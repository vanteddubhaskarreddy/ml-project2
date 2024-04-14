from flask import Flask, request, render_template

import numpy as np
import pandas as pd

from src.utils import load_object
from src.exceptions import CustomException
from src.logger import logging

from src.pipeline.predict_pipeline import PredictData, CustomData

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    else :
        data = CustomData(
            crime_rate = float(request.form['crime_rate']),
            zoned_land = float(request.form['zoned_land']),
            industry = float(request.form['industry']),
            charles_dummy_var = float(request.form['charles_dummy_var']),
            nox_conc = float(request.form['nox_conc']),
            rooms = float(request.form['rooms']),
            age = float(request.form['age']),
            distance = float(request.form['distance']),
            highways = float(request.form['highways']),
            property_tax = float(request.form['property_tax']),
            pt_ratio = float(request.form['pt_ratio']),
            black_prop = float(request.form['black_prop']),
            lower_status_popu = float(request.form['lower_status_popu'])
        )

        pred_df = data.get_data_as_data_frame()

        print("Data is: ", pred_df)

        predict_data = PredictData()
        prediction = predict_data.predict(pred_df)

        return render_template('predict.html', prediction=prediction)
    
if __name__ == "__main__":
    app.run(debug=True, port=5000)