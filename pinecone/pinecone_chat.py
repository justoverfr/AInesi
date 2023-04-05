import os
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
import datetime
import pinecone



def gpt3_embedding(content, engine='text-embedding-ada-002'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()  # fix any UNICODE errors
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def gpt3_completion(prompt, engine='text-davinci-003', temp=0.0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'RAVEN:']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            save_file('gpt3_logs/%s' % filename, prompt + '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)

if __name__ == '__main__':
    openai.api_key = open_file('openai_key.txt')
    pinecone.init(api_key=open_file('keys_pinecone.txt'), environment="us-east4-gcp")
    Ia = pinecone.Index("alnesi")
    while True:
        
        #### get user input, save it, vectorize it, save to pinecone, etc 
        a = input('\n\nUSER: ')
        timestamp = time()
        vector = gpt3_embedding(a)
        timestring = timestamp_to_datetime(timestamp)
        message = '%s: %s - %s' % ('USER', timestring, a)
        vector = gpt3_embedding(message )
        info = {'speaker': 'USER', 'time': timestamp, 'vector': vector, 'message': message, 'uuid': str(uuid4()), 'timestring': timestring}
        #### search for relevant messages, and generate a response
        resultat = Ia.query(vector = vector, top_k = 15, include_values = False)