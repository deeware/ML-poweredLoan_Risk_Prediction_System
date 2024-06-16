import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from notebook.label_encoder_mappings import Marital_status, House_Ownership, Car_Ownership, Profession

app=Flask(__name__)
## Load the model
model=pickle.load(open('notebook/model.pkl','rb'))
scalar=pickle.load(open('notebook/scaling.pkl','rb'))
@app.route('/',methods=['GET','POST'])
def home():
    
    return render_template("index.html"
                            ,Marital_status = Marital_status
                            ,House_Ownership = House_Ownership
                            ,Car_Ownership = Car_Ownership
                            ,Profession = Profession
                            ,prediction=-1)


@app.route('/predict_api',methods=['GET','POST'])
def predict_api():
    data=request.json['data']
    print(data)
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    return jsonify(output)

@app.route('/predict',methods=['GET','POST'])
def predict():
    data=[float(x) for x in list(request.form.values())[1:]]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=model.predict(final_input)[0]
    print(output)
    return render_template("index.html"
                            ,Marital_status = Marital_status
                            ,House_Ownership = House_Ownership
                            ,Car_Ownership = Car_Ownership
                            ,Profession = Profession
                            ,prediction = output)


if __name__=="__main__":
    app.run(debug=True)
   
     
