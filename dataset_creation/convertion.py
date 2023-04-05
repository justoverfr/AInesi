
# ------------------------ Convertisseur  yml -> json ------------------------ #

# import yaml
# import json

# with open("depression.yml", "r") as f:
#     data = yaml.load(f, Loader=yaml.FullLoader)

# print(data)

# json_data = []
# for item in data:
#     prompt = item[0]
#     completion = " ".join(item[1:])
#     json_data.append({"prompt": prompt, "completion": completion})

# with open("prout.json", "w") as f:
#     json.dump(json_data, f)


# ------------------------ Convertisseur  csv -> json ------------------------ #


import csv
import json

json_data = []
with open('nouveau_fichier.csv', newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        prompt = row['Questions']
        completion = row['Answers']
        json_data.append({"prompt": prompt, "completion": completion})

with open('fichier.json', 'w') as outfile:
    json.dump(json_data, outfile, ensure_ascii=False)
