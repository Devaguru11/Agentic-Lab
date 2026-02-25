from groq import Groq

client = Groq(api_key='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')  # Paste your key

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",  # Free model
    messages=[
        {"role": "user", "content": "Hello, can you help with medical data analysis?"}
    ]
)

print(response.choices[0].message.content)