import numpy as np  
import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import LabelEncoder
#create app object
app = Flask(__name__)

#load the model
model = pickle.load(open('C:\\Users\\xlor1\\Downloads\\housing predition webapp\\models\\model.pkl','rb'))
prefareaFit = pickle.load(open('C:\\Users\\xlor1\\Downloads\\housing predition webapp\\models\\prefareaFit.pkl','rb'))
hotwaterFit = pickle.load(open('C:\\Users\\xlor1\\Downloads\\housing predition webapp\\models\\hotwaterFit.pkl','rb'))
guestroomFit = pickle.load(open('C:\\Users\\xlor1\\Downloads\\housing predition webapp\\models\\guestroomFit.pkl','rb'))
mainroadFit = pickle.load(open('C:\\Users\\xlor1\\Downloads\\housing predition webapp\\models\\mainroadFit.pkl','rb'))
basementFit = pickle.load(open('C:\\Users\\xlor1\\Downloads\\housing predition webapp\\models\\basementFit.pkl','rb'))
airconditioningFit = pickle.load(open('C:\\Users\\xlor1\\Downloads\\housing predition webapp\\models\\airconditioningFit.pkl','rb'))
furnishingstatusFit = pickle.load(open('C:\\Users\\xlor1\\Downloads\\housing predition webapp\\models\\furnishingstatusFit.pkl','rb'))

le = LabelEncoder()
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    dataframe = pd.DataFrame([features])
    dataframe[4] = mainroadFit.transform(dataframe[4])
    dataframe[5] = guestroomFit.transform(dataframe[5])
    dataframe[6] = basementFit.transform(dataframe[6])
    dataframe[7] = hotwaterFit.transform(dataframe[7])
    dataframe[8] = airconditioningFit.transform(dataframe[8])
    dataframe[10] = prefareaFit.transform(dataframe[10])
    dataframe[11] = furnishingstatusFit.transform(dataframe[11])
    features = pd.DataFrame(dataframe)
    
    prediction = model.predict(features)
    return render_template('index.html', prediction_text='Estimated Price of home with given features is: {}'.format(prediction[0]))
if __name__ == '__main__':
    app.run(debug=True)