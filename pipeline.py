from transformers import pipeline 

classi = pipeline("sentiment-analysis")
resultat = classi("I dont like my friend")

print(resultat)

trans= pipeline("translation_en_to_fr")
transit = trans("My name is LEO, I have 19 years ols today !")

print(transit)


classifier = pipeline("question-answering")
question = "how old are you?"       
context = "My name is LEO, I have 19 years old today !"
res = classifier (question = question, context = context)

print(res)
