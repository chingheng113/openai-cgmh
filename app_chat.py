import os
import openai
from flask import Flask, redirect, render_template, request, url_for
import pandas as pd
import json
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")
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


@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        from_kb = False
        question = request.form["prompt"]
        query_embedding = embedder.encode(question, convert_to_tensor=True)
        kb_embedding = pd.read_csv(os.path.join('kb', 'kb_v1.csv'))
        query_scores = kb_embedding['emb'].apply(do_query, q_emb=query_embedding)
        if np.max(query_scores) > 0.7:
            content = kb_embedding['context'][np.argmax(query_scores)]
            prompt = generate_prompt(question, content)
            from_kb = True
        else:
            prompt = question
        print(prompt)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user",
                       "content": prompt
                       }]
        )
        if from_kb:
            return redirect(url_for("index", result='<The answer is from Knowledge base>' + response.choices[0].message.content))
        return redirect(url_for("index", result=response.choices[0].message.content))
    result = request.args.get("result")
    return render_template("index_chat.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8090)