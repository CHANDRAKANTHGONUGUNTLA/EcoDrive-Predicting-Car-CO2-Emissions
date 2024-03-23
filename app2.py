# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 21:04:39 2022

@author: User
"""

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

st.write("""
# CO2 Emission Prediction 
predicts the **Ammount of CO2 Emission** of a **car**.
""")

st.sidebar.header('Input Parameters')

def user_input_features():
    engine_size = st.sidebar.slider('Engine Size', 1.0, 6.8, 3.5)
    cylinder_num = st.sidebar.slider('Cylinder Number', 4, 12, 8,  step=2)
    fuelconsumption_city = st.sidebar.slider('Fuel Consumption in City', 5.3, 30.2, 15.8)
    fuelconsumption_hwy = st.sidebar.slider('Fuel Consumption in Highway', 5.1, 20.5, 12.5)
    data = {'Engine Size': engine_size,
            'Cylinder Number': cylinder_num,
            'Fuel Consumption in City': fuelconsumption_city,
            'Fuel Consumption in Highway': fuelconsumption_hwy}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Input parameters')
st.write(df)

data = pd.read_csv("C:\Users\dhruviramani\Documents\PROJECT\Copy of co2_emissions.xlsx")
data_select = data[['engine_size', 'cylinders', 'fuel_consumption_city', 'fuel_consumption_hwy', 'fuel_consumption_comb(l/100km)', 'co2_emissions']]
regr = RandomForestRegressor()
x = np.asanyarray(data_select[["engine_size", "cylinders", "fuel_consumption_city", "fuel_consumption_hwy" ]])
y = np.asanyarray(data_select[["co2_emissions"]])
regr.fit(x,y)

prediction = regr.predict(df)

st.subheader('Predicted co2 emission (in MG/L)')
st.write(prediction)
