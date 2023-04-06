import json
import pickle
from flask import Flask,request,app,jsonify,render_template,url_for
import numpy as np
import pandas as pd

app = Flask(__name__)
#load the model
regmodel = pickle.load(open('regmodel.pkl','rb'))
scalar = pickle.load(open('scaler.pkl','rb'))
@app.route('/')
def home():
    #return "Hey hai"
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html",
                           prediction_text="The House Price Prediction is {}".format(output))
app.run()
