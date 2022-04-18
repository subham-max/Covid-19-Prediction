import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

###Loading User Interface###
header= st.container()
dataset=st.container()
features=st.container()
model_training=st.container()


@st.cache
def get_data(filename):
    data=pd.read_csv(filename)

    return data

with header:
    st.title('Prediction Of COVID-19 Third Wave ')
    st.header('Depending upon the latest circumstances, our team has made an effort to analyze the upcoming pandemic')
    st.subheader('The novel coronavirus (COVID-19) that was first reported at the end of 2019 in Wuhan, China has impacted almost every aspect of life as we know it. Due to the effect of corona virus all over the world, education status also affected by this and in order to continue the course, department launched a new format.')
    st.markdown('* This Project focuses on the prediction of the future graph and show how further it can affect the world for a longer period of time.')
    st.markdown('* This report has been prepared on the project of the topic of Prediction of third wave of Covid-19 data analysis with the basis of first and second wave cases.')
    st.markdown('* The main objective of this project is to study timely trend of COVID-19in official websites of Coronavirus Source data and related to the local level websites which was followed by sorting data in excel sheets.')

####LOAD DATA####
with dataset:
    st.subheader('Countries effected by the outburst of covid-19 dataset')
    data=pd.read_csv('World_Data_Set.csv',sep=',')
    #st.bar_chart(data.head())
    data=data[['Days','cases']]
    st.line_chart(data.head(50))
    #print('-'*30);print('HEAD');print('-'*30)
    #print(data.head())

with features:
    st.header('Features of this Project-')
    st.markdown('* **1:** This Project predicts the upcoming impact of covid-19.')
    st.markdown('* **2:** This Project shows you the accuracy rate of the model.')
    st.markdown('* **3:** This Project takes the user input and predicts the number of people may get effected.')

with model_training:
    st.header('Taking Input from User in number of days and showing the output')
    sel_cols,disp_cols=st.columns(2)
    days=sel_cols.slider('Number of days should be predicted?',min_value=1,max_value=100,value=1,step=1)

####PREPARE DATA####
#print('-'*30);print('PREPARE DATA');print('-'*30)
x=np.array(data['Days']).reshape(-1,1)
y=np.array(data['cases']).reshape(-1,1)
plt.plot(y,'-m')
#plt.show()
polyFeat=PolynomialFeatures(degree=2)
x=polyFeat.fit_transform(x)
#print(x)



####TRAINING DATA####
#print('-'*30);print('TRAINING DATA');print('-'*30)
model=linear_model.LinearRegression()
model.fit(x,y)
accuracy=model.score(x,y)
disp_cols.text('The Accuracy of the Model:')
disp_cols.write(f'{round(accuracy*85,3)}%')
y0=model.predict(x)



####PREDICTION####
#days=int(input("Enter number of days:"))
#print('-'*30);print('PREDICTION');print('-'*30)
#print(f'Prediction - Cases after {days} days:',end='')
disp_cols.text('Cases predicted: ')
disp_cols.write((round(int(model.predict(polyFeat.fit_transform([[365+days]])))/1000000,2),'Million'))

x1=np.array(list(range(1,365+days))).reshape(-1,1)
y1=model.predict(polyFeat.fit_transform(x1))
plt.plot(y1,'--r')
plt.plot(y0,'--b')
st.subheader('Graphical Representation')
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot()
st.subheader('In the above graph we highlight-The Implementation of the machine learning model')
st.markdown('*  Red dots indicates future prediction' )
st.markdown('*  Violet line showcase the outbreak by a graphical representation' )
st.markdown('*  Blue dots indicates a model graph created by analyzing the data sets and predict the upcoming outbreak' )










