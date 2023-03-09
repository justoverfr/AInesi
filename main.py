from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import langcodes

import torch

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# ---------------------------------------------------------------------------- #
#                              Hugging Face models                             #
# ---------------------------------------------------------------------------- #
language_detection_model = "ivanlau/language-detection-fine-tuned-on-xlm-roberta-base"

conversation_model = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(conversation_model)
model = AutoModelForCausalLM.from_pretrained(conversation_model)

# ---------------------------------------------------------------------------- #
#                                   Fonctions                                  #
# ---------------------------------------------------------------------------- #


def get_language(text):
    tokenizer = AutoTokenizer.from_pretrained(language_detection_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        language_detection_model)

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)

    label_id = torch.argmax(outputs.logits, axis=1).item()
    labels = model.config.id2label
    language_code = labels[label_id]
    return language_code


def get_language_iso(language_name):
    language_code = langcodes.find(language_name).language
    return language_code


def translate_to_english(text, language_iso):
    if language_iso == "en":
        return text
    else:
        translator = pipeline(
            "translation", model=f"Helsinki-NLP/opus-mt-{language_iso}-en")
        return translator(text)[0]['translation_text']


def translate_to_language(text, target_language_iso):
    if target_language_iso != "en":
        translator = pipeline(
            "translation", model=f"Helsinki-NLP/opus-mt-en-{target_language_iso}")
        return translator(text)[0]['translation_text']

    else:
        return text


def get_emotions(text):
    emotions_analysis = pipeline(
        "text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    emotions_array = emotions_analysis(text)[0]

    # Cacul du score de bonne humeur
    good_score = next(
        (item['score'] for item in emotions_array if item['label'] == 'joy'), None)

    # Cacul du score de mauvaise humeur
    disgust_score = next(
        (item['score'] for item in emotions_array if item['label'] == 'disgust'), None)
    fear_score = next(
        (item['score'] for item in emotions_array if item['label'] == 'fear'), None)
    anger_score = next(
        (item['score'] for item in emotions_array if item['label'] == 'anger'), None)
    sadness_score = next(
        (item['score'] for item in emotions_array if item['label'] == 'sadness'), None)

    bad_score = sum(
        filter(None, [disgust_score, fear_score, anger_score, sadness_score]))

    # Calcul du score moyen de bonne humeur
    set_user_happiness(good_score, bad_score)


def set_user_happiness(good_score, bad_score, alpha=0.2):
    global user_happiness

    # Calcul du niveau d'humeur en fonction du score de bonne humeur et du score de mauvaise humeur et du niveau d'humeur précédent
    happiness_score = alpha * user_happiness + \
        (1 - alpha) * (good_score - bad_score)

    # On s'assure que le nouveau niveau d'humeur est compris entre 0 et 1
    happiness_score = max(0, min(happiness_score, 1))

    # On met à jour le niveau d'humeur
    user_happiness = happiness_score


def get_response(text, ai_language, step):
    global new_user_input_ids, bot_input_ids, chat_history_ids
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(
        text + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat(
        [chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(
        bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    ai_message = tokenizer.decode(
        chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return translate_to_language(ai_message, ai_language)


user_happiness = 1
if __name__ == '__main__':

    step = 0
    print("===== AInesi =====")
    while True:
        user_message = input("Vous : ")

        if user_message == "quit":
            break

        language = get_language(user_message)
        language_iso = get_language_iso(language)

        en_message = translate_to_english(user_message, language_iso)

        print(f"Texte traduit en anglais : {en_message}")  # TODO A retirer

        get_emotions(en_message)
        print(f"Score de bonne humeur : {user_happiness * 100 :.2f}%")

        ai_response = get_response(en_message, language_iso, step)
        print(f"AInesi : {ai_response}")

        step += 1
