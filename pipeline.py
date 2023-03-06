'''from transformers import pipeline 

#analyse de sentiment
classi = pipeline("sentiment-analysis")
resultat = classi("I dont like my friend")

print(resultat)

#traduction
trans= pipeline("translation_en_to_fr")
transit = trans("My name is LEO, I am 19 years old today !")

print(transit)

#question réponses
classifier = pipeline("question-answering")
question = "how old are you?"       
context = "My name is LEO, I am 19 years old today !"
res = classifier (question = question, context = context)

print(res)'''
