from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import langcodes

import torch

# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

language_detection_model = "ivanlau/language-detection-fine-tuned-on-xlm-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(language_detection_model)
model = AutoModelForSequenceClassification.from_pretrained(
    language_detection_model)


def get_language(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)

    label_id = torch.argmax(outputs.logits, axis=1).item()
    labels = model.config.id2label
    language_code = labels[label_id]
    return language_code


def get_language_iso(language_name):
    language_code = langcodes.find(language_name).language
    print(language_code)
    return language_code


def translate_to_english(text):
    language = get_language(text)
    language_iso = get_language_iso(language)

    if language_iso == "en":
        return text
    else:
        translator = pipeline(
            "translation", model=f"Helsinki-NLP/opus-mt-{language_iso}-en")
        return translator(text)[0]['translation_text']


def get_sentiment(text):
    sentiment_analysis = pipeline("sentiment-analysis")
    return sentiment_analysis(text)[0]


def rate_sentiment(text, sentiment):
    if sentiment["label"] == "NEGATIVE":
        classifier = pipeline("zero-shot-classification")

        depression_analysis = classifier(text, candidate_labels=[
            "depressive", "positive", "dislike", "neutral", "desire"])

        print(depression_analysis)

        depression_id = depression_analysis["labels"].index("depressive")
        depression_level = depression_analysis["scores"][depression_id]

        print(f"Le niveau de dépression est de {depression_level}")

        if depression_level >= 0.1:
            print("Ce message semble être dépressif")

        else:
            print("Ce message n'est pas dépressif")


if __name__ == '__main__':
    text_input = input("Entrez un message : ")

    en_text = translate_to_english(text_input)
    sentiment = get_sentiment(en_text)

    rate_sentiment(en_text, sentiment)

    print(f"Sentiment : {sentiment}")
