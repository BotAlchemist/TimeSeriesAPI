from flask import Flask, render_template, request, url_for, redirect
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import jsonify
from azure.storage.blob import BlobClient
import os
import pickle
import json

vader = SentimentIntensityAnalyzer()



app= Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/result', methods=['POST'])
def result():
    raw_text= [str(x) for x in request.form.values()]
    text= [raw_text]

    sentiment_result= vader.polarity_scores(text[0][0])
    negative_value = round(sentiment_result['neg']*100,2)
    neutral_value = round(sentiment_result['neu']*100,2)
    positive_value = round(sentiment_result['pos']*100,2)
    compound_value = sentiment_result['compound']
    if compound_value >= 0.05:
        overall_value= "Positive"
    elif compound_value <= -0.05:
        overall_value= "Negative"
    else:
        overall_value= "Neutral"

    return render_template('home.html', sentiment_text= text[0][0],
                           sentiment_result= sentiment_result,
                           negative_value= negative_value,
                           neutral_value= neutral_value,
                           positive_value= positive_value,
                           compound_value= abs(round(compound_value,2)),
                           overall_value=overall_value
                           )


@app.route('/arima-api/<int:n>')
def arima_api(n):
    no_of_steps= n
     
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
        
    
    

if __name__ == '__main__':
    app.run(debug=True)
