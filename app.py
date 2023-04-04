from transformers import XLNetForSequenceClassification, XLNetTokenizer, BertForSequenceClassification, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer
import torch
from typing import Dict
import gradio as gr


model = BertForSequenceClassification.from_pretrained(
    "./Personality_detection_Classification_Save/", num_labels=5)  # =num_labels)
tokenizer = BertTokenizer.from_pretrained(
    './Personality_detection_Classification_Save/', do_lower_case=True)
model.config.label2id = {
    "Extroversion": 0,
    "Neuroticism": 1,
    "Agreeableness": 2,
    "Conscientiousness": 3,
    "Openness": 4,
}

model.config.id2label = {
    "0": "Extroversion",
    "1": "Neuroticism",
    "2": "Agreeableness",
    "3": "Conscientiousness",
    "4": "Openness", }


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
        dict1 = tokenizer.encode_plus(
            Preprocess_part1, max_length=1024, padding=True, truncation=True)
        dict2 = tokenizer.encode_plus(
            Preprocess_part2, max_length=1024, padding=True, truncation=True)
        dict_custom['input_ids'] = [dict1['input_ids'], dict1['input_ids']]
        dict_custom['token_type_ids'] = [
            dict1['token_type_ids'], dict1['token_type_ids']]
        dict_custom['attention_mask'] = [
            dict1['attention_mask'], dict1['attention_mask']]
        outs = model(torch.tensor(dict_custom['input_ids']), token_type_ids=None, attention_mask=torch.tensor(
            dict_custom['attention_mask']))
        b_logit_pred = outs[0]
        pred_label = torch.sigmoid(b_logit_pred)
        ret = {
            "Extroversion": float(pred_label[0][0]),
            "Neuroticism": float(pred_label[0][1]),
            "Agreeableness": float(pred_label[0][2]),
            "Conscientiousness": float(pred_label[0][3]),
            "Openness": float(pred_label[0][4]), }
        return ret


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

Fotter = (

    "<center>Copyright &copy; 2023 - All Rights Reserved</center>"
)

css_path = "./style.css"

app = gr.Interface(
    Personality_Detection_from_reviews_submitted,
    inputs=model_input,
    outputs=model_output,
    examples=examples,
    title=title,
    description=description,
    css=css_path
)

app.launch(debug=True, show_error=False)
