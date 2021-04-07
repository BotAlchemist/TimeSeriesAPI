# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 18:44:51 2021

@author: Sumit
"""

from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
from azure.storage.blob import BlobClient
import os
import pickle
import json


app = Flask(__name__)


@app.route('/api/', methods=['POST', 'GET'])
def arimaforecast():
    data = request.get_json()  
    
    blob = BlobClient(account_url="https://sumitfilestorage.blob.core.windows.net",
                      container_name="model-container",
                      blob_name="finalized_model.sav",
                      credential="4AIaVvS+u6WC7mpgSnyPwr2LjWZwF9GwSQEXZD/+/b7m+BZsPdbx7k1csueZv514YiDzm6zWf3tTqgOdEzc6nA==")
    
    with open("finalized_model.sav", "wb") as f:
        model = blob.download_blob()
        model.readinto(f)
        
    
        
    loaded_model = pickle.load(open("finalized_model.sav", 'rb')) 
    
    prediction = loaded_model.forecast(data)
    prediction= np.array2string(prediction[0])

    return jsonify(prediction)




if __name__ == '__main__':
    app.run(debug=True)
