"""
llm_client.py
=============

One tiny interface -- ``llm.invoke(system, user, role)`` -- behind which the
rest of the project is completely LLM-agnostic.

Backends
--------
* ``LangChainOpenAI``  : ``prompt | ChatOpenAI | StrOutputParser`` (preferred)
* ``OpenAISDK``        : plain ``openai`` client, used if LangChain is missing
* ``MockLLM``          : offline simulator (see mock_llm.py), used when there is
                         no API key so the app still runs out of the box

``get_llm()`` picks the right one and never raises because of a missing key
or a missing package; it just falls back to the simulator and records why in
``llm.notice`` so the UI can tell the user.

The ``role`` argument ("generate" | "gradient" | "optimize") is ignored by the
real backends. It only exists so the offline simulator knows which of the three
jobs it is being asked to do.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "gpt-4o-mini"


def _timeout_s() -> float:
    """Per-request timeout. Local models on CPU can be slow, hence the generous default."""
    try:
        return float(os.environ.get("OPENAI_TIMEOUT", "180"))
    except ValueError:
        return 180.0


class LLMError(RuntimeError):
    """Raised for any problem talking to the model (auth, network, bad answer...)."""


class BaseLLM:
    """Common bookkeeping shared by all backends."""

    name: str = "base"
    is_simulated: bool = False
    notice: str = ""

    def __init__(self) -> None:
        self.call_count = 0

    def invoke(self, system: str, user: str, role: str = "generate") -> str:
        self.call_count += 1
        return self._invoke(system, user, role)

    def _invoke(self, system: str, user: str, role: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


def _supports_temperature(model: str) -> bool:
    """Reasoning-model families reject an explicit temperature."""
    return not model.lower().startswith(("o1", "o3", "o4", "gpt-5"))


class LangChainOpenAI(BaseLLM):
    """LangChain chain: ChatPromptTemplate | ChatOpenAI | StrOutputParser."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None) -> None:
        super().__init__()
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        kwargs: dict = {"model": model, "api_key": api_key, "timeout": _timeout_s(), "max_retries": 2}
        if base_url:
            kwargs["base_url"] = base_url
        if _supports_temperature(model):
            kwargs["temperature"] = 0.0
        chat = ChatOpenAI(**kwargs)

        # The prompt text is passed as *variables*, so braces inside an
        # optimised prompt or a SQL string are never interpreted as template slots.
        template = ChatPromptTemplate.from_messages([("system", "{system}"), ("human", "{user}")])
        self._chain = template | chat | StrOutputParser()
        self.name = f"LangChain + {model}"

    def _invoke(self, system: str, user: str, role: str) -> str:
        try:
            return self._chain.invoke({"system": system, "user": user})
        except Exception as exc:  # network, auth, rate limit, ...
            raise LLMError(f"LLM call failed ({type(exc).__name__}): {exc}") from exc


class OpenAISDK(BaseLLM):
    """Fallback that uses the plain ``openai`` package."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None) -> None:
        super().__init__()
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=_timeout_s(), max_retries=2)
        self._model = model
        self.name = f"OpenAI SDK + {model}"

    def _invoke(self, system: str, user: str, role: str) -> str:
        kwargs: dict = {}
        if _supports_temperature(self._model):
            kwargs["temperature"] = 0.0
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                **kwargs,
            )
        except Exception as exc:
            raise LLMError(f"LLM call failed ({type(exc).__name__}): {exc}") from exc
        return response.choices[0].message.content or ""


class CachedLLM(BaseLLM):
    """
    Wraps a live backend with a persistent on-disk response cache (SQLite).

    Requests are made at temperature 0, so an identical request (same model, system
    prompt and user message) is answered from the cache. That makes long benchmark runs
    *resumable* (an interrupted run continues where it stopped), makes repeated runs
    free, and saves a lot of time with slow local models. Errors are never cached.
    """

    def __init__(self, inner: BaseLLM, path: str | os.PathLike) -> None:
        super().__init__()
        self.inner = inner
        self.path = Path(path)
        self.name = inner.name
        self.is_simulated = inner.is_simulated
        self.notice = inner.notice
        self.hits = 0
        self.misses = 0
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT NOT NULL)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.path), timeout=10)

    def _key(self, system: str, user: str) -> str:
        return hashlib.sha256(f"{self.inner.name}\x00{system}\x00{user}".encode("utf-8")).hexdigest()

    def _invoke(self, system: str, user: str, role: str) -> str:
        key = self._key(system, user)
        conn = self._connect()
        try:
            row = conn.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
            if row is not None:
                self.hits += 1
                return row[0]
            answer = self.inner.invoke(system, user, role)
            self.misses += 1
            if answer and answer.strip():
                conn.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, answer))
                conn.commit()
            return answer
        finally:
            conn.close()

    def entries(self) -> int:
        """Return the number of cached responses."""
        # Use the same database path and schema initialized in __init__.
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def clear_cache(self) -> int:
        """Delete all cached responses; returns how many were removed."""
        with self._connect() as conn:
            removed = conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0]
            conn.execute("DELETE FROM responses")
            conn.commit()
        self.hits = self.misses = 0
        return removed


def _with_cache(llm: BaseLLM) -> BaseLLM:
    """Wrap ``llm`` in a disk cache unless disabled with TEXTGRAD_SQL_CACHE=0 (or the folder is unwritable)."""
    setting = os.environ.get("TEXTGRAD_SQL_CACHE", "1").strip()
    if setting.lower() in ("0", "false", "off", "no"):
        return llm
    candidates = []
    if setting.lower() not in ("1", "true", "on", "yes", ""):
        candidates.append(Path(setting))
    candidates += [Path(__file__).resolve().parent / ".llm_cache.sqlite",
                   Path(tempfile.gettempdir()) / "textgrad_sql_llm_cache.sqlite"]
    for path in candidates:
        try:
            cached = CachedLLM(llm, path)
            cached.notice = f"{llm.notice} Responses are cached in {path.name} (resumable runs)."
            return cached
        except (OSError, sqlite3.Error):
            continue
    return llm


def get_llm(
    prefer: str = "auto",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseLLM:
    """
    Build the best available backend.

    prefer = "auto" -> real model if a key is available, otherwise demo mode
    prefer = "demo" -> always the offline simulator
    prefer = "live" -> real model; falls back to demo (with a notice) if it cannot be built
    """
    # A .env file next to the project is picked up if python-dotenv is installed.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    from mock_llm import MockLLM

    # Support both generic OpenAI-compatible settings and a dedicated Groq key.
    # If GROQ_API_KEY is present, use Groq's endpoint unless another base URL
    # has explicitly been supplied.
    groq_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    key = (api_key or groq_key or os.environ.get("OPENAI_API_KEY") or "").strip()

    configured_base_url = (
        base_url or os.environ.get("OPENAI_BASE_URL") or ""
    ).strip()
    if configured_base_url:
        base_url = configured_base_url
    elif groq_key and not api_key:
        base_url = "https://api.groq.com/openai/v1"
    else:
        base_url = None

    default_model = "openai/gpt-oss-120b" if groq_key and not api_key else DEFAULT_MODEL
    model = (model or os.environ.get("OPENAI_MODEL") or default_model).strip()

    if prefer == "demo":
        llm = MockLLM()
        llm.notice = "Demo mode was selected: answers come from the offline simulator, not a real model."
        return llm

    if not key:
        llm = MockLLM()
        llm.notice = (
            "No GROQ_API_KEY or OPENAI_API_KEY found, so the app is running in demo mode "
            "with an offline simulator. Set an API key in your .env file to use a real model."
        )
        return llm

    errors: list[str] = []
    for backend in (LangChainOpenAI, OpenAISDK):
        try:
            llm = backend(model, key, base_url)
            llm.notice = f"Live mode: {llm.name}."
            return _with_cache(llm)
        except ImportError as exc:
            errors.append(f"{backend.__name__}: {exc}")
        except Exception as exc:
            errors.append(f"{backend.__name__}: {type(exc).__name__}: {exc}")

    llm = MockLLM()
    llm.notice = (
        "An API key was found but no OpenAI client could be created "
        f"({'; '.join(errors)}). Run `pip install -r requirements.txt`. Falling back to demo mode."
    )
    return llm


@dataclass
class LLMStatus:
    """Small value object handy for displaying the current backend."""

    mode: str  # "live" | "demo"
    name: str
    notice: str

    @classmethod
    def of(cls, llm: BaseLLM) -> "LLMStatus":
        return cls("demo" if llm.is_simulated else "live", llm.name, llm.notice)
