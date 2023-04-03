import openai

openai.api_key = "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"

response = openai.Completion.create(engine="text-davinci-003", prompt="This is a test", temperature=0.6,
                                    max_tokens=5,
                                    top_p=1,
                                    frequency_penalty=0,
                                    presence_penalty=0.6,
                                    stop=["\n", " Human:", " AI:"])

print(response)
