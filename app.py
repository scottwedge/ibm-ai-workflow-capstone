from flask import Flask, jsonify, request
import joblib
import socket
import json
import pandas as pd
import os
import logging
import modelling

app = Flask(__name__)
host = '0.0.0.0'
port = 5000


@app.route("/")
def home():

    html = "<h3>Hello {name}!</h3>" \
           "<b>Hostname:</b> {hostname}<br/>"
    return html.format(name=os.getenv("NAME", "world"), hostname=socket.gethostname())

@app.route('/train', methods=['GET','POST'])
def train():
    ## input checking
    if not request.json:
        print("ERROR: API (train): did not receive request data")
        return jsonify([])

    query = request.json
    query = pd.DataFrame(query)
    
    if len(query.shape) == 1:
         query = query.reshape(1, -1)

    trained_model = modelling.train_model(query) # this appends new data to the old one before training
    pickle.dump(trained_model, open(saved_model, 'wb')) # this will save the new trained model

    return(jsonify(trained_model))     

@app.route('/predict', methods=['GET','POST'])
def predict():
    ## input checking
    if not request.json:
        print("ERROR: API (predict): did not receive request data")
        return jsonify([])

    query = request.json
    query = pd.DataFrame(query)
    
    if len(query.shape) == 1:
        query = query.reshape(1, -1)

    y_pred = modelling.predict(query)
    
    return(jsonify(y_pred.tolist()))   


if __name__ == '__main__':
    saved_model = 'models/trained-model.sav'
    model = joblib.load(saved_model)
    
    logging.basicConfig(filename='predictions.log',
        format='%(asctime)s %(message)s' ,
        datefmt='%m/%d/%Y %I:%M:%S %p')

    try:
        logging.info('prediction: ' + str(y_pred))
    except:
        pass
        
    app.run(host=host, port=port, debug=True)