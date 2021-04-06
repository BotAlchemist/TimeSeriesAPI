# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 12:44:47 2021

@author: Sumit
"""

from flask import Flask, jsonify
from azure.storage.blob import BlobClient
import os
import pickle
import json

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/arima-api/<int:n>')
def arima_api(n):
    no_of_steps= n
    
    print(no_of_steps)
     
    blob = BlobClient(account_url="https://sumitfilestorage.blob.core.windows.net",
                      container_name="model-container",
                      blob_name="finalized_model.sav",
                      credential="4AIaVvS+u6WC7mpgSnyPwr2LjWZwF9GwSQEXZD/+/b7m+BZsPdbx7k1csueZv514YiDzm6zWf3tTqgOdEzc6nA==")
    
    with open("finalized_model.sav", "wb") as f:
        model = blob.download_blob()
        model.readinto(f)
        
    
        
    loaded_model = pickle.load(open("finalized_model.sav", 'rb'))   
    output = loaded_model.forecast(no_of_steps)
    output= output[0]
    output= [round(i,2) for i in output]
    
    
    return str(output)
    
@app.route('/arima-meta')
def arima_meta():
    blob = BlobClient(account_url="https://sumitfilestorage.blob.core.windows.net",
                      container_name="model-container",
                      blob_name="meta_data.txt",
                      credential="4AIaVvS+u6WC7mpgSnyPwr2LjWZwF9GwSQEXZD/+/b7m+BZsPdbx7k1csueZv514YiDzm6zWf3tTqgOdEzc6nA==")
    
    with open("meta_data.txt", "wb") as f:
        data = blob.download_blob()
        data.readinto(f)
        
    with open('meta_data.txt') as json_file:
        data = json.load(json_file)
        return  str(data['Metrics'])
        
    
    


if __name__== "__main__":
    app.run(debug=True)
