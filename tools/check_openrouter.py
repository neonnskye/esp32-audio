import os
import time

from openai import OpenAI

api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

start = time.time()

completion = client.chat.completions.create(
    extra_headers={
        "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
        "X-OpenRouter-Title": "<YOUR_SITE_NAME>",  # Optional. Site title for rankings on openrouter.ai.
    },
    model="google/gemini-2.5-flash-lite",
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
)

print(completion.choices[0].message.content)

end = time.time()
print(f"Time taken: {end - start} seconds")
