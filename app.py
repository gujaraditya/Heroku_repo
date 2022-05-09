import pickle

from flask import Flask,request,jsonify,render_template
import numpy as np
import pickle
app=Flask(__name__)
model=pickle.load(open('LR_model.pkl','rb'))

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    get_features=[float(x) for x in request.form.values()]
    features=[np.array(get_features)]
    prediction= model.predict(features)
    return render_template('index.html',prediction_text='The Species of the flower is : {}'.format(prediction[0]))

if __name__=='__main__':
    app.run(debug=True) # will update to the web dynamically



