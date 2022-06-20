# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier
st.set_page_config(layout='centered', page_title='Penguin Classification', page_icon='logo.png')
# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv("penguin.csv")

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)


# Create a function that accepts 'island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g' and 'sex' as inputs and returns the species name.
@st.cache()
def prediction(model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
  species = model.predict([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]])
  species = species[0]
  #'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2
  if species == 0:
    return "Adelie"
  elif species == 1:
    return "Chinstrap"
  else:
    return "Gentoo"  

# Design the App

# Add title widget
st.title("Penguin Species Prediction App")  

# Add 4 sliders for numeric values  - 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g' and store the value returned by them in 5 separate variables.
b_len = st.slider("Bill Length in mm", float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()))
b_dep = st.slider("Bill Depth in mm", float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()))
f_len = st.slider("Flipper Length in mm", float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()))
body_mass = st.slider("Body Mass in gms", float(df['body_mass_g'].min()), float(df['body_mass_g'].max()))

# Add 2 dropdown for categorical values  - 'sex' and 'island' and convert them to numeric
gen = st.selectbox('Gender', ('Male', 'Female'))

if gen == 'Male':
  gen = 0
else:
  gen = 1


isl = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
if isl == 'Biscoe':
  isl = 0
elif isl == 'Dream':
  isl = 1
else:
  isl = 2  


classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))

# When the 'Predict' button is clicked, check which classifier is chosen and call the 'prediction()' function.
# Store the predicted value in the 'species_type' variable accuracy score of the model in the 'score' variable. 
# Print the values of 'species_type' and 'score' variables using the 'st.text()' function.
if st.sidebar.button("Predict"):
  if classifier == 'Support Vector Machine':
    species_type = prediction(svc_model, isl, b_len, b_dep, f_len, body_mass, gen)
    score = svc_score

  elif classifier =='Logistic Regression':
    species_type = prediction(log_reg, isl, b_len, b_dep, f_len, body_mass, gen)
    score = log_reg_score

  else:
    species_type = prediction(rf_clf, isl, b_len, b_dep, f_len, body_mass, gen)
    score = rf_clf_score
  
  st.write("Species predicted:", species_type)
  st.write("Accuracy score of this model is:", score)
  st.write('Thank you!')
