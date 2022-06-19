import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)
df.head()
df = df.dropna()
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})
df['sex'] = df['sex'].map({'Male':0,'Female':1})
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})
xtr, xts, ytr, yts= train_test_split(df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']], df['label'], test_size = 0.33, random_state = 42)
svc = SVC(kernel = 'linear')
svc.fit(xtr, ytr)
lreg = LogisticRegression()
lreg.fit(xtr, ytr)
rfc = RandomForestClassifier(n_jobs = -1)
rfc.fit(xtr, ytr)
def prediction(model, island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex):
	return model.predict(island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex)
st.set_page_config(layout='centered', page_title='Penguin Classification', page_icon='logo.png')
st.header('Penguin Classification')
st.title('Set parameters')
bl = st.slider('Bill Length (in mm)', float(df['bill_length_mm'].min()), float(df['bill_length_mm'].max()))
bd = st.slider('Bill Depth (in mm)', float(df['bill_depth_mm'].min()), float(df['bill_depth_mm'].max()))
fl = st.slider('Flipper Length (in mm)', float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()))
bm = st.slider('Body Mass (in g)', float(df['body_mass_g'].min()), float(df['body_mass_g'].max()))
gen = st.selectbox('Gender of the penguin', ('Male', 'Female'))
isl = st.selectbox('Island in which the penguin lives', ('Dream', 'Biscoe', 'Torgersen'))
m = st.selectbox('Model for classififcation', ('RandomForestClassifier', 'LogisticRegression', 'SVC'))
if m == 'RandomForestClassifier':
	m = rfc
elif m=='SVC':
	m = svc
else:
	m = lreg
if isl == 'Dream':
	isl = 0
elif isl=='Biscoe':
	isl = 1
else:
	isl = 2
if gen == 'Dream':
	gen = 0
elif gen=='Biscoe':
	gen = 1
else:
	gen = 2
if st.button('Predict the class of the penguin!'):
	st.write(f'The penguin is predicted to be of class {prediction(m, isl, bl, bd, fl, bm, gen)}.\nThe accruracy of the model used for prediction is {m.score(xtr, ytr)}')
	st.subtitle('Thank you!')