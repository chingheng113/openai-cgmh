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



def do_query(row, q_emb):
    row = json.loads(row)
    corpus_emb = torch.as_tensor(row).cuda()
    cos_scores = util.cos_sim(q_emb, corpus_emb)[0]
    highest_scores = torch.topk(cos_scores, k=1).values.item()
    return highest_scores


def get_ans(prompt, pfx):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0,
        logprobs=1
    )
    ans = response.choices[0].text
    logprob = response.choices[0].logprobs
    token_logprobs = logprob.token_logprobs
    # pd.DataFrame(token_logprobs).to_csv(pfx+'.csv', index=False)
    return np.mean(token_logprobs), ans


def doQA(q_str, kb, cnt):
    query_embedding = embedder.encode(q_str, convert_to_tensor=True)
    query_scores = kb['emb'].apply(do_query, q_emb=query_embedding)
    if np.max(query_scores) > 0.7:
        score_inxs = np.argpartition(query_scores, -4)[-4:]
        content_id = kb['context_idx'][score_inxs]
        content_list = cnt['context'][cnt.context_idx.isin(content_id)]
        i = 0
        new_content_list = []
        for c in content_list:
            prompt = generate_prompt(q_str, c)
            log_prob, ans = get_ans(prompt, str(query_scores[score_inxs.iloc[i]]))
            if log_prob > -0.1:
                new_content_list.append(ans)
            i += 1
        if len(new_content_list) == 0:
            print('I don\'t know')
        elif len(new_content_list) == 1:
            print(new_content_list[0])
        else:
            new_prompt = generate_prompt(q_str, ''.join(new_content_list))
            log_prob, ans = get_ans(new_prompt, '')
            print('**' + str(log_prob))
            print('**' + ans)
    else:
        print('pass')



questions = ['What are the three categories of benign breast lesions?',
             'what is stroke?',
             'how are you?',
             'How many categories of benign breast lesions?',
             'What are the eleven categories of benign breast lesions?',
             'What are the three categories of benign breast lesions and What is atypical hyperplasia?'
             ]

kb_embedding = pd.read_csv(os.path.join('kb', 'breast_cancer_embeddings.csv'))
contents = pd.read_csv(os.path.join('kb', 'breast_cancer_context.csv'))
for qq in questions:
    print(qq)
    doQA(qq, kb_embedding, contents)
    print('###')

print('done')