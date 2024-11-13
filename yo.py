from openai import OpenAI

client = OpenAI(api_key="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")

  # Replace with your actual key

def test_openai_api():
    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, world!"}],
        max_tokens=5)
        print("OpenAI response:", response.choices[0].message.content)
    except Exception as e:
        print("Error:", e)

test_openai_api()
