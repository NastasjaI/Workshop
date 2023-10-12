#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[ ]:


import pandas as pd
import lightgbm as lgb
from flask import Flask, request, jsonify
import joblib

loaded_model = joblib.load('lgbm_model.pkl')

app = Flask("default")

@app.route("/predict", methods=["POST"])

def predict():
    X = request.get_json()
    preds = loaded_model.predict(pd.DataFrame(X, index=[0]))
    result = {"default_proba": preds.tolist()}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8989)

