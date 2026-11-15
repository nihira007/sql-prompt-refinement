
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)

api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError("API_KEY not found in .env file")

client = Groq(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": "Say hello in one sentence."}
        ],
        temperature=0.2,
    )

    print(response.choices[0].message.content)

except Exception as e:
    print("Groq API error:", e)
