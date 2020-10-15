  
import flask
import pickle
import pandas as pd

# Use pickle to load in the pre-trained model
with open(f'model/Vignesh.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        # Extract the input
        Pregnancies = flask.request.form['Pregnancies']
        Glucose = flask.request.form['Glucose']
        BloodPressure = flask.request.form['BloodPressure']
        BMI = flask.request.form['BMI']
        Age = flask.request.form['Age']


        # Make DataFrame for model
        input_variables = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, BMI, Age]],
                                       columns=['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'Age'],
                                       dtype=float,
                                       index=['input'])

        # Get the model's prediction
        predictions = model.predict(input_variables)[0]
    
        # Render the form again, but add in the prediction and remind user
        # of the values they input before
        return flask.render_template('main.html',result=predictions)
from waitress import serve
import os
if __name__ == '__main__':
    app.run()