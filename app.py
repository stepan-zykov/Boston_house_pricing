import pickle
from flask import (Flask, request, app, jsonify, url_for, 
                   render_template, redirect, flash, session)
import numpy as np
import pandas as pd

app = Flask(__name__)
# load the model and the scalar
reg_model = pickle.load(open('reg_model.pkl', 'rb'))
scalar = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = reg_model.predict(new_data)
    print(output)
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = reg_model.predict(final_input)[0]
    return render_template("home.html", prediction_text=f'The house price prediction is {output}')
if __name__ == '__main__':
    app.run(debug=True)