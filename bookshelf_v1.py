import os
import openai
from flask import Flask, redirect, render_template, request, url_for
import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np


openai.api_key =
model_engine = "gpt-3.5-turbo-0301"
embedder = SentenceTransformer('all-MiniLM-L6-v2')


def generate_prompt(question, kb_content):
    role_prompt = 'as a physician, '
    task_prompt = 'your primary goal is to answer patients\' questions to the best of your ability.\n'
    content_prompt = 'START CONTEXT\n'+kb_content+'\nEND CONTEXT\n'
    question_prompt = 'patient: '+question+'\nphysician:'
    final_prompt = role_prompt+task_prompt+content_prompt+question_prompt
    return final_prompt


def get_gpt_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']


def do_query(row, q_emb):
    row = json.loads(row)
    corpus_emb = torch.as_tensor(row).cuda()
    cos_scores = util.cos_sim(q_emb, corpus_emb)[0]
    highest_scores = torch.topk(cos_scores, k=1).values.item()
    return highest_scores


def do_gpt_query(row, q_emb):
    q_emb = torch.as_tensor(q_emb).cuda()
    row = json.loads(row)
    corpus_emb = torch.as_tensor(row).cuda()
    cos_scores = util.cos_sim(q_emb, corpus_emb)[0]
    highest_scores = torch.topk(cos_scores, k=1).values.item()
    return highest_scores


question = 'What are the three categories of benign lesions of the breast?'


kb_embedding = pd.read_csv(os.path.join('kb', 'breast_cancer_embeddings.csv'))
contents = pd.read_csv(os.path.join('kb', 'breast_cancer_context.csv'))

# BERT
query_embedding = embedder.encode(question, convert_to_tensor=True)
query_scores = kb_embedding['emb'].apply(do_query, q_emb=query_embedding)
if np.max(query_scores) > 0.7:
    content_id = kb_embedding['context_idx'][np.argmax(query_scores)]
    content = contents['context'][contents.context_idx == content_id].values[0]
    prompt = generate_prompt(question, content)
    from_kb = True
else:
    prompt = question

response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user",
                       "content": prompt
                       }]
        )
ans = response.choices[0].message.content

print(ans)


# GPT
gpt_embedding = get_gpt_embedding(question)
query_scores = kb_embedding['questions_embedding'].apply(do_gpt_query, q_emb=gpt_embedding)
if np.max(query_scores) > 0.7:
    content_id = kb_embedding['context_idx'][np.argmax(query_scores)]
    content = contents['context'][contents.context_idx == content_id].values[0]
    prompt = generate_prompt(question, content)
    from_kb = True
else:
    prompt = question

response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user",
                       "content": prompt
                       }]
        )
ans = response.choices[0].message.content
print(ans)

print('done')