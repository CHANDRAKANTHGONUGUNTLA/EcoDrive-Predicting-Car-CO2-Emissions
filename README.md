# Business Objective:
The fundamental goal here is to model the CO2 emissions as a function of several car engine features.

# Data Set Details:
The file contains the data for this example. Here the number of variables (columns) is 12, and the number of instances (rows) is 7385. In that way, this problem has the 12 following variables:
- make: Car brand under study.
- model: Specific model of the car.
- vehicle_class: Car body type of the car.
- engine_size: Size of the car engine, in Litres.
- cylinders: Number of cylinders.
- transmission: Type of transmission, including "A" for Automatic, "AM" for Automated manual, "AS" for Automatic with select shift, "AV" for Continuously variable, and "M" for Manual.
- fuel_type: Type of fuel used, including "X" for Regular gasoline, "Z" for Premium gasoline, "D" for Diesel, "E" for Ethanol (E85), and "N" for Natural gas.
- fuel_consumption_city: City fuel consumption ratings, in litres per 100 kilometres.
- fuel_consumption_hwy: Highway fuel consumption ratings, in litres per 100 kilometres.
- fuel_consumption_comb(l/100km): Combined fuel consumption rating (55% city, 45% highway), in L/100 km.
- fuel_consumption_comb(mpg): Combined fuel consumption rating (55% city, 45% highway), in miles per gallon (mpg).
- co2_emissions: Tailpipe emissions of carbon dioxide for combined city and highway driving, in grams per kilometer.

## CODE BREAKDOWN
### Part 1: Importing Necessary Libraries
```python
import pandas as pd 
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
```
### Explanation:
Imports the pandas library for data manipulation, numpy for numerical operations, matplotlib for plotting, and seaborn for enhanced visualization capabilities.
### Part 2: Loading the Dataset
```python
Data = pd.read_csv("co2_emissions.csv", sep=";")
```
### Explanation:

Loads the dataset from a CSV file named "co2_emissions.csv" using the pd.read_csv() function.
The sep=";" parameter specifies that the delimiter in the CSV file is a semicolon.
### Part 3: Data Exploration and Preprocessing
```python
Data.head(20)
Data.info()
print(Data['make'].unique())
print(Data['model'].unique())
# Handling categorical variables
Data['transmission'] = np.where(Data['transmission'].isin(['A4','A5','A6','A7','A8','A9','A10']),'Automatic',Data['transmission'])
# More categorical handling...
print(Data['fuel_type'].value_counts())
```
### Explanation:

Displays the first 20 rows of the dataset and provides information about the dataset's structure.
Checks unique values for certain categorical variables like 'make', 'model', etc., to understand the data distribution.
Performs categorical variable handling, such as grouping similar sub-categories into broader categories.
### Part 4: Feature Engineering
```python
Data_v = pd.get_dummies(Data['fuel_type'], prefix='Fuel', drop_first=True)
Data.v = pd.get_dummies(Data["transmission"], drop_first=True)
df = [Data, Data_v, Data.v]
data = pd.concat(df, axis=1)
data.drop(['fuel_type'], inplace=True, axis=1)
data.drop(['transmission'], inplace=True, axis=1)
```
### Explanation:

Performs one-hot encoding on categorical variables.
Concatenates the one-hot encoded features with the original dataset.
Drops the original categorical columns as they are no longer needed.
### Part 5: Feature Selection
```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
ranked_feature = SelectKBest(score_func=chi2, k='all')
ordered_feature = ranked_feature.fit(A, B)
```
### Explanation:

Uses SelectKBest with Chi-square test to select the most important features.
### Part 6: Splitting the Data
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(A, B, test_size=0.2, random_state=42)
```
### Explanation:

Splits the dataset into training and testing sets using the train_test_split() function.
### Part 7: Data Scaling
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
### Explanation:

Performs feature scaling using standardization to bring the features to the same scale.
### Part 8: Model Training and Evaluation
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
### Explanation:

Initializes and trains a Linear Regression model on the training data.
Makes predictions on the testing data.
### Part 9: Overall Performance Analysis
```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('R^2:', r2_score(y_test, y_pred))
```
### Explanation:
Evaluates the performance of the model using metrics such as RMSE and R-squared score.
### Result:
<img width="648" alt="Screenshot 2024-03-22 at 8 18 21 PM" src="https://github.com/CHANDRAKANTHGONUGUNTLA/EcoDrive-Predicting-Car-CO2-Emissions/assets/97879005/3ed33141-ffe7-4a2b-8841-f89a4c82d111">

#### Overall Decision Tree Regression model given the best results.

## DEPLOYMENT
### Part 10: Setting Up Streamlit App and User Input
```python
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

st.write("""
# CO2 Emission Prediction 
predicts the **Amount of CO2 Emission** of a **car**.
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
```
### Explanation:

This part sets up the Streamlit application and defines the user interface.
It allows users to input parameters such as engine size, cylinder number, fuel consumption in the city, and fuel consumption on the highway using sliders on the sidebar.
The user input is then displayed in a subheader.
### Part 11: Loading Data and Model Training
```python
data = pd.read_csv("C:\Users\dhruviramani\Documents\PROJECT\Copy of co2_emissions.xlsx")
data_select = data[['engine_size', 'cylinders', 'fuel_consumption_city', 'fuel_consumption_hwy', 'fuel_consumption_comb(l/100km)', 'co2_emissions']]
regr = RandomForestRegressor()
x = np.asanyarray(data_select[["engine_size", "cylinders", "fuel_consumption_city", "fuel_consumption_hwy" ]])
y = np.asanyarray(data_select[["co2_emissions"]])
regr.fit(x,y)
```
### Explanation:

This part loads the dataset from a CSV file and selects relevant columns for model training.
It then initializes a RandomForestRegressor model and trains it using the selected features (X) and the target variable (y).
### Part 12: Making Predictions
```python
prediction = regr.predict(df)
```
### Explanation:

This part makes predictions using the trained RandomForestRegressor model on the user input data (df), which contains the selected features.
### Part 13: Displaying Predictions
```python
st.subheader('Predicted co2 emission (in MG/L)')
st.write(prediction)
```
### Explanation:

Finally, this part displays the predicted CO2 emission values obtained from the model in the Streamlit app.
### OUTPUT:
<img width="915" alt="image" src="https://github.com/CHANDRAKANTHGONUGUNTLA/EcoDrive-Predicting-Car-CO2-Emissions/assets/97879005/912f28fe-9781-419b-984b-ea1ef3bd20af">
