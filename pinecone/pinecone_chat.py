import os
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time, sleep
from uuid import uuid4
import datetime
import pinecone

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    # fix any UNICODE errors
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def gpt3_completion(prompt, engine='text-davinci-003', temp=0.0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['USER:', 'RAVEN:']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII', errors='ignore').decode()
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
            save_file('gpt3_logs/%s' % filename, prompt +
                      '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)

def load_conversation(results):
    result = list()
    for m in results['matches']:
        info = load_json('nexus/%s.json' % m['id'])
        result.append(info)
    # sort them all chronologically
    ordered = sorted(result, key = lambda d: d['time'], reverse=False)
    messages = [i['message'] for i in ordered]
    return '\n'.join(messages).strip()


if __name__ == '__main__':
    convo_lenght = 30
    openai.api_key = open_file('openai_keys.txt')
    pinecone.init(api_key=open_file('keys_pinecone.txt'),
                  environment="us-east4-gcp")
    Ia = pinecone.Index("alnesi")
    while True:

        # get user input, save it, vectorize it, save to pinecone, etc

        playload = list()
        a = input('\n\nUSER: ')
        timestamp = time()
        vector = gpt3_embedding(a)
        timestring = timestamp_to_datetime(timestamp)
        message = '%s: %s - %s' % ('USER', timestring, a)
        vector = gpt3_embedding(message)
        metadata = {'speaker': 'USER', 'time': timestamp,'message': message, 'timestring': timestring, 'uuid': unique_id}
        unique_id = str(uuid4())
        save_json('nexus/%s.json' % unique_id, metadata)
        playload.append((unique_id, vector))
        # search for relevant messages, and generate a response
        results = Ia.query(vector=vector, top_k=convo_lenght)
        conversation = load_conversation(results)
        prompt = open_file('prompt_response.txt').replace('<<CONVERSATION>>', conversation).replace('<<MESSAGE>>', a)
        
        output = gpt3_completion(prompt)
        timestamp = time()
        vector = gpt3_embedding(output)
        unique_id = str(uuid4())
        timestring = timestamp_to_datetime(timestamp)
        message = '%s: %s - %s' % ('RAVEN', timestring, output)
        info = {'speaker': 'alnesi', 'time': timestamp, 'vector': vector, 'message': message, 'uuid': str(uuid4()), 'timestring': timestring}
        metadata = {'speaker': 'alnesi', 'time': timestamp,'message': message, 'timestring': timestring, 'uuid': unique_id}
        unique_id = str(uuid4())
        save_json('nexus/%s.json' % unique_id, metadata)
        playload.append((unique_id, vector))
        Ia.upsert(playload)
        print('\n\alnesi: %s' % output)
        
