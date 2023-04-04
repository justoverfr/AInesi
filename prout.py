import csv
import unicodedata

def replace_weird_characters(text):
    # Convertir le texte en NFC pour normaliser la représentation des caractères
    text = unicodedata.normalize("NFC", text)
    # Remplacer les caractères bizarres par leur équivalent UTF-8
    text = text.encode("utf-8", "ignore").decode("utf-8")
    return text

with open("Mental_Health_FAQ.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    data = [row for row in reader]

new_data = []
for row in data:
    new_row = [replace_weird_characters(col) for col in row]
    new_data.append(new_row)

with open("nouveau_fichier.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(new_data)
