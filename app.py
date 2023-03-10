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
    if len(model_input) < 20:
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
    "Input text here (Note: This model is trained to classify Big Five Personality Traits From Expository text features)", show_label=False)
model_output = gr.Label(" Big-Five personality traits Result", num_top_classes=6, show_label=True,
                        label="Big-Five personality traits Labels assigned to this text based on its features")
examples = [
    ("Well, here we go with the stream-of-consciousness essay. I used to do things like this in high school sometimes.",
     "They were pretty interesting, but I often find myself with a lack of things to say. ",
     "I normally consider myself someone who gets straight to the point. I wonder if I should hit enter any time to send this back to the front",
     "Maybe I'll fix it later. My friend is playing guitar in my room now. Sort of playing anyway.",
     "More like messing with it. He's still learning. There's a drawing on the wall next to me. "
     ),
    ("An open keyboard and buttons to push. The thing finally worked and I need not use periods, commas, and all those things.",
        "Double space after a period. We can't help it. I put spaces between my words and I do my happy little assignment of jibber-jabber.",
        "Babble babble babble for 20 relaxing minutes and I feel silly and grammatically incorrect. I am linked to an unknown reader.",
        "A graduate student with an absurd job. I type. I jabber and I think about dinoflagellates. About sunflower crosses and about ",
        "the fiberglass that has to be added to my lips via clove cigarettes and I think about things that I shouldn't be thinking.",
        "I know I shouldn't be thinking. or writing let's say/  So I don't. Thoughts don't solidify. They lodge in the back. behind my tongue maybe.",
     ),

    ("My favorite aspect of debate would actually be --this all gets back to a time when we were assigned to write a bill that we would take to a fake model ",
        "united nations conference and we would have to present a bill that we wanted to be passed- in fact, my partner and I rarely wanted the bills we proposed to be ",
        "passed, but we just wanted people to have to argue against them, in most cases we would try to make our bills interesting or at least darkly satirical, ",
        "so that the only arguments that could be made against them would be based on moral rationalization rather than common reason- the moral debates would most ",
        "likely get everyone interested and could be defeated by one who was willing not to be moral- none of our bills ever passed-As I write this I find that I am ",
        "often losing my train of thought but I don't believe that that is how I usually think- as a result of the confines of this experiment I am discovering that",
        "I am thinking more quickly than I normally do and I can't explain why that is other to keep typing, however, when I am normally thinking, I still try to think",
     ),

    ("slowly and articulately so as not to speak something that makes me look ignorant-this is said mostly to point to out possible flaws in the ways of tracing thoughts .",
        "Now in fact I a running out of things to say before I finish, which is still about seven minutes away- I'd like to apologize for the many spelling errors that ",
        "are sure to be found in his assignment-  don't mean the errors that are natural such as words that I just don't know the spelling of but rather, I mean the words ",
        "that look as if they have been written by an idiot because I am not a very talented typist and my fingers are slipping over the keys, I would go back and fix these ",
        "errors but that seems contradictory to the nature of the assignment  2:53 was the time at which I am writing this I am also realizing that occasionally there is ",
        "no clear and concise thought n my head which I can write down or there are just so many thoughts that I can not possibly transfer them onto paper at the rate at which ",
        "they are passing through- I hate leaving the impression with anyone that I am ignorant and I think that is the main reason I dislike this assignment because I don't ",
        "see how anyone can read this and not see exactly that- it is my hope that at least everyone will appear ignorant and then at least I will be on even ground.",
        "I also hate writing this to a professor of psychology because I am sure it is analyzed more than is necessary- if this assignment is done honestly then you could probably ",
        "just talk to someone and get just as many honest answers- well -I've just hit nineteen minutes and I suppose that last sentence is just as good a place to finish off as any where. ")

]

title = "<center><a href=\"https://thoucentric.com/\"><img src='https://thoucentric.com/wp-content/themes/cevian-child/assets/img/Thoucentric-Logo.png' alt='Thoucentric-Logo'></a></center><br>Big Five Personality Traits Detection From Expository text features"
description = ("<br><br>In traditional machine learning, it can be challenging to train an accurate model if there is a lack of labeled data specific to the task or domain of interest. Transfer learning offers a way to address this issue by utilizing the pre-existing labeled data from a similar task or domain to improve model performance. By transferring knowledge learned from one task to another, transfer learning enables us to overcome the limitations posed by a shortage of labeled data, and to train more effective models even in data-scarce scenarios. We try to store this knowledge gained in solving the source task in the source domain and applying it to our problem of interest. In this work, I have utilized Transfer Learning utilizing BERT BASE UNCASED model to fine-tune on Big-Five Personality traits Dataset.")

Fotter = (

    "<center>Copyright &copy; 2023 <a href=\"https://thoucentric.com/\">Thoucentric</a>. All Rights Reserved</center>"
)

app = gr.Interface(
    Personality_Detection_from_reviews_submitted,
    inputs=model_input,
    outputs=model_output,
    examples=examples,
    title=title,
    description=description,
    article=Fotter,
    allow_flagging='never',
    analytics_enabled=False,
)

app.launch(show_error=False)
