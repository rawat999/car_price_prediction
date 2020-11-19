#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:36:34 2020

@author: prem
"""

import numpy as np
import pickle
#import pandas as pd
#from flasgger import Swagger
import streamlit as st 
#import sklearn.ensemble._forest

#from PIL import Image

#app=Flask(__name__)
#Swagger(app)

pickle_in = open("car_price_rf_model.pkl","rb")
classifier=pickle.load(pickle_in)
feature = pickle.load(open("car_feature.pkl","rb"))

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_car_price(data):
    
    """Let's PREDICT THE PRICE OF THE CARS 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name:Volume(cm3)
        in: query
        type: number
        required: true
      - name: Year
        in: query
        type: number
        required: true
      - name: condition
        in: query
        type: string
        required: true
      - name: fuel type
        in: query
        type: string
        required: true
      - name:color
        in: query
        type: string
        required: true
      - name: transmission
        in: query
        type:string
        required: true
      - name:drive_unit
        in: query
        type:string
        required: true
      - name:segment
        in: query
        type: string
        required: true
    responses:
        200:
            description: The output values
        
    """
    
    prediction=classifier.predict([data])
    print(prediction)
    return np.exp(prediction)


def handle_condition(param):
    condition = feature[2:5]
    arr = np.zeros((1,len(condition)),dtype=int).tolist()[0]
    try:
        index = condition.index(param)
        arr[index] = 1
        return arr
    except ValueError:
        return arr


def handle_fuel(param):
    fuel_type = feature[5:7]
    arr = np.zeros((1,len(fuel_type)),dtype=int).tolist()[0]
    try:
        index = fuel_type.index(param)
        arr[index] = 1
        return arr
    except ValueError:
        return arr

def handle_color(param):
    color = feature[7:20]
    arr = np.zeros((1,len(color)),dtype=int).tolist()[0]
    try:
        index = color.index(param)
        arr[index] = 1
        return arr
    except ValueError:
        return arr


def handle_transmission(trans):
    transmission = feature[20:22]
    arr = np.zeros((1,len(transmission)),dtype=int).tolist()[0]
    try:
        index = transmission.index(trans)
        arr[index] = 1
        return arr
    except ValueError:
        return arr

def handle_drive_unit(param):
    drive_unit = feature[22:26]
    arr = np.zeros((1,len(drive_unit)),dtype=int).tolist()[0]
    try:
        index = drive_unit.index(param)
        arr[index] = 1
        return arr
    except ValueError:
        return arr

def handle_segment(param):
    segment = feature[26:]
    arr = np.zeros((1,len(segment)),dtype=int).tolist()[0]
    try:
        index = segment.index(param)
        arr[index] = 1
        return arr
    except ValueError:
        return arr
    
def preparing_data(volume,car_age,condition,fuel_type,color,transmission,drive_unit,segment):
    data_list = []
    data_list += [volume]
    data_list += [car_age]
    data_list += condition
    data_list += fuel_type
    data_list += color
    data_list += transmission
    data_list += drive_unit
    data_list += segment
    return data_list
    

def main():
    st.title("Car Price Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Car Price Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    volume = st.text_input("Volume(cm3)","Type Here")
    year = st.text_input("Year","Type Here")
    condition = st.selectbox('Condition',feature[2:5])
    fuel_type = st.selectbox("Fuel Type",feature[5:7])
    color = st.selectbox("Color",feature[7:20])
    transmission = st.selectbox("Transmission",feature[20:22])
    drive_unit = st.selectbox("Drive Unit",feature[22:26])
    segment = st.selectbox("Segment",feature[26:])
    result=""
    if st.button("Predict"):
        car_age = 2020 - int(year)
        condition = handle_condition(condition)
        fuel_type = handle_fuel(fuel_type)
        color = handle_color(color)
        transmission = handle_transmission(transmission)
        drive_unit = handle_drive_unit(drive_unit)
        segment = handle_segment(segment)
        prepared_data = preparing_data(volume,car_age,condition,fuel_type,color,transmission,drive_unit,segment)
        result=predict_car_price(prepared_data)
    st.success('Predicted Price of the Car is {}'.format(result))
    if st.button("About"):
        st.text("Built by Prem Singh Rawat")
        st.text("Release Date: 19 December, 2020")

if __name__=='__main__':
    main()
    