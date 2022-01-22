from typing import List
from tqdm import tqdm
import demoji
from emoji import demojize, UNICODE_EMOJI
from vncorenlp import VnCoreNLP
import io
from flask import Flask, render_template, request, make_response, Response
from flask_cors import CORS, cross_origin
from werkzeug.exceptions import BadRequest
import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.svm import SVC, LinearSVC
import pickle

import logging

logger = logging.getLogger()
logger.level = logging.ERROR


MAX_LEN = 256
dictOfModels = {}


app = Flask(__name__)
cors = CORS(app, supports_credentials=True)
app.config['CORS_HEADERS'] = 'Content-Type'


phobert_base = AutoModel.from_pretrained("vinai/phobert-base")
tokenizer_base = AutoTokenizer.from_pretrained(
    "vinai/phobert-base", use_fast=False)
rdrsegmenter = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar",
                         annotators="wseg", max_heap_size='-Xmx500m')


def tokenize_sentence(sent: str) -> str:
    if sent.__class__ == str:
        list_demojized_tokens = []
        for token in sent.split(' '):
            # Need a better way to extract emoji
            if len(token) == 1:
                list_demojized_tokens.append(demojize(token))
            else:
                list_demojized_tokens.append(token)

        tokenized_sents = rdrsegmenter.tokenize(
            ' '.join(list_demojized_tokens))
        res = []
        for sent in tokenized_sents:
            res.append(' '.join(sent))

        return ' '.join(res)
# Route
#


def phobert_phi(input_ids: List[int]):
    """
        Convert sentences into vectors
    """
    X = torch.tensor([input_ids])

    with torch.no_grad():
        reps = phobert_base(X)
        return reps.last_hidden_state.squeeze(0).numpy()


def phobert_classifier_phi(text, func: str = None):
    """
        Combines resulting vectors
    """
    reps = phobert_phi(text)

    if func == None:
        return reps[0]
    if func == 'mean':
        return reps.mean(axis=0)


def get_prediction(text, model):
    # Tokenize
    tokenized = tokenize_sentence(text)
    #
    input = tokenizer_base.encode(tokenized, add_special_tokens=True)
    #
    v_input = [phobert_classifier_phi(input, func='mean')]
    y_pred = model.predict(v_input)
    return y_pred


@app.route('/text', methods=['POST'])
@cross_origin(supports_credentials=True)
def predict_text():
    # Get text
    # Checking

    text = request.form.get("text")
    if text == None:
        raise BadRequest("Missing file parameter!")
    result = get_prediction(
        text, dictOfModels[request.form.get("model_choice")])[0]
    print('result', result)
    response = json.dumps({'result': result}, indent=4)
    return Response(response, mimetype='application/json')


# Entry
if __name__ == '__main__':
    print('Starting webservice...')
    # Getting directory containing models from command args (or default 'models_train')
    models_directory = 'models_train'
    if len(sys.argv) > 1:
        models_directory = sys.argv[1]
    print(f'Watching for models under {models_directory}...')
    for r, d, f in os.walk(models_directory):
        for file in f:
            if ".pkl" in file:
                # example: file = "model1.pt"
                # the path of each model: os.path.join(r, file)
                model_name = os.path.splitext(file)[0]
                model_path = os.path.join(r, file)
                model_file = open(model_path, 'rb')
                print(
                    f'Loading model {model_path} with path {model_path}...')
                dictOfModels[model_name] = pickle.load(model_file)
                # you would obtain: dictOfModels = {"model1" : model1 , etc}

    print(
        f'Server now running on ')

    # starting app
    app.run(debug=True, host='0.0.0.0', port=5005)
