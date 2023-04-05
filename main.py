from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BertForSequenceClassification, BertTokenizer
from typing import Dict
import gradio as gr

import torch
import openai

# ---------------------------------------------------------------------------- #
#                              Hugging Face models                             #
# ---------------------------------------------------------------------------- #

# Conversation
openai_model = "davinci"
openai_api_key = "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"

openai.api_key = openai_api_key

# Emotions
emotions_model_name = "j-hartmann/emotion-english-distilroberta-base"

emotions_analysis = pipeline(
    "text-classification", model=emotions_model_name, return_all_scores=True)

# Personnalité
personality_model_name = "./Personality_detection_Classification_Save/"

personality_model = BertForSequenceClassification.from_pretrained(
    personality_model_name, num_labels=5)  # =num_labels)
personality_tokenizer = BertTokenizer.from_pretrained(
    personality_model_name, do_lower_case=True)

personality_model.config.label2id = {
    "Extroversion": 0,
    "Neuroticism": 1,
    "Agreeableness": 2,
    "Conscientiousness": 3,
    "Openness": 4,
}

personality_model.config.id2label = {
    "0": "Extroversion",
    "1": "Neuroticism",
    "2": "Agreeableness",
    "3": "Conscientiousness",
    "4": "Openness", }

# ---------------------------------------------------------------------------- #
#                                   Fonctions                                  #
# ---------------------------------------------------------------------------- #


def get_emotions(text):
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


def set_user_happiness(good_score, bad_score, alpha=0.7):
    global user_happiness

    # Calcul du niveau d'humeur en fonction du score de bonne humeur et du score de mauvaise humeur et du niveau d'humeur précédent
    happiness_score = alpha * user_happiness + \
        (1 - alpha) * (good_score - bad_score)

    # On s'assure que le nouveau niveau d'humeur est compris entre 0 et 1
    happiness_score = max(0, min(happiness_score, 1))

    # On met à jour le niveau d'humeur
    user_happiness = happiness_score


def Personality_Detection_from_reviews_submitted(model_input: str) -> Dict[str, float]:
    if len(model_input) < 10:
        ret = {
            "Extroversion": float(0),
            "Neuroticism": float(0),
            "Agreeableness": float(0),
            "Conscientiousness": float(0),
            "Openness": float(0), }
        return ret

    else:
        # Encoding input data
        dict_custom = {}
        Preprocess_part1 = model_input[:len(model_input)]
        Preprocess_part2 = model_input[len(model_input):]
        dict1 = personality_tokenizer.encode_plus(
            Preprocess_part1, max_length=1024, padding=True, truncation=True)
        dict2 = personality_tokenizer.encode_plus(
            Preprocess_part2, max_length=1024, padding=True, truncation=True)
        dict_custom['input_ids'] = [dict1['input_ids'], dict1['input_ids']]
        dict_custom['token_type_ids'] = [
            dict1['token_type_ids'], dict1['token_type_ids']]
        dict_custom['attention_mask'] = [
            dict1['attention_mask'], dict1['attention_mask']]
        outs = personality_model(torch.tensor(dict_custom['input_ids']), token_type_ids=None, attention_mask=torch.tensor(
            dict_custom['attention_mask']))
        b_logit_pred = outs[0]
        pred_label = torch.sigmoid(b_logit_pred)
        ret = {
            "Extroversion": float(pred_label[0][0]),
            "Neuroticism": float(pred_label[0][1]),
            "Agreeableness": float(pred_label[0][2]),
            "Conscientiousness": float(pred_label[0][3]),
            "Openness": float(pred_label[0][4]), }

        # return ret
        return set_user_personality(ret)


def set_user_personality(personality_scores):
    global user_personality

    # On met à jour le niveau de chaque trait de personnalité
    result = {}
    for trait in user_personality.keys():
        result[trait] = (float)(user_personality[trait] +
                                personality_scores[trait]) / 2

    user_personality = result


def get_response(message):
    conversation_history = f"User: {message}\nIA:"
    response_generation = openai.Completion.create(
        engine="davinci",
        prompt=conversation_history,
        # # Ajustez la température pour contrôler la créativité des réponses (0.7 est une valeur recommandée)
        # temperature=0.7,
        # max_tokens=150,         # Limitez le nombre de tokens dans la réponse générée
        # # Utilisez la méthode "nucleus sampling" pour sélectionner les réponses (1 signifie que toutes les réponses possibles seront prises en compte)
        # top_p=1,
        # # Ajustez la pénalité de fréquence pour éviter des réponses trop fréquentes ou trop rares
        # frequency_penalty=0,
        # # Ajustez la pénalité de présence pour éviter la répétition des tokens dans la réponse
        # presence_penalty=0,
        # Indiquez les caractères de fin pour arrêter la génération de texte (ici, on arrête après un saut de ligne)
        stop=["\n"],
    )

    ai_message = response_generation.choices[0].text.strip()

    return ai_message


def send_message(user_message):
    get_emotions(user_message)
    print(f"Score de bonne humeur : {user_happiness * 100 :.2f}%")

    Personality_Detection_from_reviews_submitted(user_message)
    print(f"Personnalité : {user_personality}")

    response = get_response(user_message)
    print(f"Réponse : {response}")
    return response


user_happiness = 1
user_personality = {'Extroversion': 0.5, 'Neuroticism': 0.5,
                    'Agreeableness': 0.5, 'Conscientiousness': 0.5, 'Openness': 0.5}

model_input = gr.Textbox(
    "Bonjour !", show_label=False)
model_output = gr.Label("AInesi", num_top_classes=6, show_label=True,
                        label="Réponse de AInesi")
examples = [
    ("J'ai mal au coeur")
]

title = "<center><h1>AInesi</h1></center>"
description = (
    "<br><br>AInesi est une Intelligence Artificielle de soutien émotionnelle et psychologique")
footer = (
    "<center>Copyright &copy; 2023 - All Rights Reserved</center>"
)

css_path = "./style.css"

app = gr.Interface(
    fn=send_message,
    inputs=model_input,
    outputs=model_output,
    examples=examples,
    title=title,
    description=description,
    css=css_path
)


if __name__ == '__main__':

    # app.launch(debug=True, show_error=False, share=True)
    message = input("Message : ")
    sauce = send_message(message)

    # print("===== AInesi =====")
    # while True:
    #     user_message = input("Vous : ")

    #     if user_message == "quit":
    #         break

    #     language = get_language(user_message)
    #     language_iso = get_language_iso(language)

    #     en_message = translate_to_english(user_message, language_iso)

    #     # print(f"Texte traduit en anglais : {en_message}")

    #     get_emotions(en_message)
    #     print(f"Score de bonne humeur : {user_happiness * 100 :.2f}%")

    #     Personality_Detection_from_reviews_submitted(en_message)
    #     print(f"Personnalité : {user_personality}")

    #     ai_response = get_response(en_message, language_iso, step)
    #     print(f"AInesi : {ai_response}")

    #     step += 1
