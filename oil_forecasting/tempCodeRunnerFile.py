from flask import Flask, render_template, request
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

app = Flask(__name__)

# Define a function to train the SARIMAX model and make predictions
def train_sarimax_model(data, p, d, q, a, b, c, m):
    # Select a single column as the target variable (endogenous variable)
    endog = data['value']  # Replace 'column_name' with the actual column name you want to use

    # Create and fit the SARIMAX model
    model = SARIMAX(endog, order=(p, d, q), seasonal_order=(a, b, c, m))
    model_fit = model.fit()

    # Make predictions
    predictions = model_fit.predict(start=len(endog), end=len(endog) + 49)  # Predict the next 50 days

    return predictions

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    p = int(request.form['p'])
    d = int(request.form['d'])
    q = int(request.form['q'])
    a = int(request.form['a'])
    b = int(request.form['b'])
    c = int(request.form['c'])
    m = int(request.form['m'])

    # Read the time series data
    data = pd.read_csv('crude.csv')  # Replace 'crude_oil_prices.csv' with your actual data file
    try:
        value = request.form['value']
        # Rest of the code...
    except KeyError as e:
        return f"KeyError: {e}"

    try:
        # Select a single column as the target variable (endogenous variable)
        endog = data[value]  # Replace 'column_name' with the actual column name you want to use
    except KeyError as e:
        return f"KeyError: {e}"

    # Train the SARIMAX model and make predictions
    predictions = train_sarimax_model(endog, p, d, q, a, b, c, m)

    # Pass the predictions to the result page
    return render_template('result.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
