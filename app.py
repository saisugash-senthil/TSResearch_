from flask import Flask, render_template,url_for,flash,redirect,request
import sklearn
import pickle
from flask import Flask, render_template
import json
from werkzeug.utils import secure_filename
import pandas as pd
import os
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pyodbc




with open ("LagLength.json" ,'r') as e:
    lagvar = e.read()
LagLength = json.loads(lagvar)

app = Flask(__name__)
model = pickle.load(open('tseriesmodel.pkl','rb'))


@app.route('/',methods=['GET','POST'])
def home():
    return render_template('home.html',text_boxes=LagLength)

@app.route('/modelpage',methods=['POST'])
def modelpage():
    return render_template('modelpage.html',text_boxes=LagLength)

@app.route('/uploadpage',methods=['POST'])
def uploadpage():
    return render_template('uploadpage.html',text_boxes=LagLength)

@app.route('/make_predict',methods=['POST'])
def make_predict():
    plot_html = generate_conf()
    if request.method == 'POST':
        if 'csvFile' in request.files:
            print('csv file detected.')
            csv_file = request.files['csvFile']
            if csv_file.filename !='':
                filename = os.path.join(os.getcwd(),secure_filename(csv_file.filename))
                csv_file.save(filename)
                df = pd.read_csv(filename)
                print(df)
                csv_values=df.values.flatten().tolist()
                prediction_result = model.predict([csv_values])
                # os.remove(filename)
                return render_template('result.html', prediction_result=prediction_result,text_boxes=LagLength,plot_html=plot_html)
    parameters = [int(request.form[f'Day{i}']) for i in range(1, LagLength + 1)]
    prediction_result = model.predict([parameters])
    return render_template('result.html', prediction_result=prediction_result,text_boxes=LagLength,plot_html =plot_html)
'''The following functions and code is to plot the confidence intervals on the webpage'''

def find_confidence_interval_gaussian(confidenceLevel,sampleMean,sampleStandardError,error,ForecastedData):
    degrees_freedom = len(error)-1  #degree of freedom = sample size-1
    #sampleMean = np.mean(error)    #sample mean
    #sampleStandardError = st.sem(error)  #sample standard error
    confidenceInterval95 = st.norm.interval(confidence=confidenceLevel, loc=sampleMean, scale=sampleStandardError)
    lower95=pd.DataFrame(ForecastedData+confidenceInterval95[0])
    upper95=pd.DataFrame(ForecastedData+confidenceInterval95[1])
    df = pd.concat([lower95, upper95], axis = 1)
    return df


def find_Prediction_Interval_Evaluation(pi, Original, Forecasted, ci):
    # Computation of Prediction Interval Coverage Probability (PICP)
    count = 0
    cc = pd.DataFrame()
    for i in range(0, Original.shape[0], 1):
        dat = Original.iloc[i, 0]
        if dat >= pi.iloc[i, 0] and dat <= pi.iloc[i, 1]:
            count += 1
    PICP = (1 / Original.shape[0]) * count * 100

    # Computation of Prediction Interval Normalized Average Width (PINAW)
    sum = 0
    for i in range(0, pi.shape[0], 1):
        sum += pi.iloc[i, 1] - pi.iloc[i, 0]
    A = np.max(Original) - np.min(Original)
    PINAW = (1 / (A * pi.shape[0])) * sum

    # Computation of Accumulated Width Deviation (AWD)
    sum = 0
    for i in range(0, Original.shape[0], 1):
        dat = Original.iloc[i, 0]
        if dat >= pi.iloc[i, 0] and dat <= pi.iloc[i, 1]:
            sum += 0
        elif dat < pi.iloc[i, 0]:
            sum += (pi.iloc[i, 0] - dat) / (pi.iloc[i, 1] - pi.iloc[i, 0])
        else:
            sum += (dat - pi.iloc[i, 1]) / (pi.iloc[i, 1] - pi.iloc[i, 0])
    AWD = (1 / Original.shape[0]) * sum

    # Computation of Average Coverage Error (ACE)=PICP-PINC
    ACE = PICP - ci
    return PICP, PINAW, AWD, ACE
def generate_conf():
    my_data = pd.read_csv("C:\\Users\\ssais\\Documents\\Research under professors\\Sibarama-FacAd\\ConfidenceInt.csv")
    error = pd.DataFrame(my_data.iloc[310:424,2])
    ForecastedData = pd.DataFrame(my_data.iloc[310:424,1])
    Original = pd.DataFrame(my_data.iloc[310:424,0])

    location=0
    Scale=5
    conf95=find_confidence_interval_gaussian(0.95,location,Scale,error,ForecastedData)
    conf90=find_confidence_interval_gaussian(0.90,location,Scale,error,ForecastedData)
    conf80=find_confidence_interval_gaussian(0.80,location,Scale,error,ForecastedData)
    conf70=find_confidence_interval_gaussian(0.70,location,Scale,error,ForecastedData)
    conf60=find_confidence_interval_gaussian(0.60,location,Scale,error,ForecastedData)
    fig, ax = plt.subplots(1,1)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    ax.fill_between(conf90.index,conf90.iloc[:,0],conf90.iloc[:,1], color='#cf0000',label='90%')
    ax.fill_between(conf80.index,conf80.iloc[:,0],conf80.iloc[:,1], color='#af0000',label='80%')
    ax.fill_between(conf90.index,conf70.iloc[:,0],conf70.iloc[:,1], color='#8f0000',label='70%')
    ax.fill_between(conf80.index,conf60.iloc[:,0],conf60.iloc[:,1], color='#6f0000',label='60%')
    ax.plot(ForecastedData, linewidth=3,label='Forecasts')
    ax.plot(ForecastedData, linewidth=3,label='True Values',color='black', ls=':')
    ax.legend(loc="best")
    ax.set(xlabel='Time / Month', ylabel='Monthly Crude Oil Price')
    PIEvaluation=pd.DataFrame(np.zeros((5,4)))
    PIEvaluation.iloc[1,0], PIEvaluation.iloc[1,1], PIEvaluation.iloc[1,2], PIEvaluation.iloc[1,3]=find_Prediction_Interval_Evaluation(conf90,Original,ForecastedData,90)
    PIEvaluation.iloc[2,0], PIEvaluation.iloc[2,1], PIEvaluation.iloc[2,2], PIEvaluation.iloc[2,3]=find_Prediction_Interval_Evaluation(conf80,Original,ForecastedData,80)
    PIEvaluation.iloc[3,0], PIEvaluation.iloc[3,1], PIEvaluation.iloc[3,2], PIEvaluation.iloc[3,3]=find_Prediction_Interval_Evaluation(conf70,Original,ForecastedData,70)
    PIEvaluation.iloc[4,0], PIEvaluation.iloc[4,1], PIEvaluation.iloc[4,2], PIEvaluation.iloc[4,3]=find_Prediction_Interval_Evaluation(conf60,Original,ForecastedData,60)
    img_buf = BytesIO()
    plt.savefig(img_buf,format='png')
    img_buf.seek(0)

    image_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
    img_tag = f'<img src="data:image/png;base64,{image_base64}" alt="plot">'
    return img_tag
