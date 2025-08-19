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



from groq import Groq

client = Groq(
    api_key="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"
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    model="llama3-8b-8192",  # or other available models
)

print(chat_completion.choices[0].message.content)
