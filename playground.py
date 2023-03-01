import openai
import sys

openai.api_key = "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"
model_engine = "text-davinci-003"
prompt = "what is hypertension"

try:
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.9
    )
    data = response["choices"][0]["text"].strip()
    print(data)
except Exception as msg:
    print(msg)
    sys.exit()