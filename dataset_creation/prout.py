import json

with open('intents.json', 'r') as f:
    data = json.load(f)

output = []
for d in data:
    for p in d['patterns']:
        for r in d['responses']:
            output.append({
                'prompt': p,
                'completion': r
            })

with open('output.json', 'w') as f:
    json.dump(output, f)
