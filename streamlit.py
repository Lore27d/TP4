import streamlit as st
import pandas as pd
#import seaborn as sns
#from matplotlib import pyplot as plt
import pickle

st.title("Predicción de Temperatura")
st.write("Escribiendo...")

dataframe = pd.DataFrame(np.random.randn(10, 20),columns=[f'col {i}' for i in range(20)])

st.dataframe(dataframe.style.highlight_max(axis=0))
st.table(dataframe)

map_data = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],columns=['lat', 'lon'])
st.map(map_data)



with open('./model_arima.pkl', 'rb') as f_arima:
        modelo_arima = pickle.load(f_arima)

with open('./model_est.pkl', 'rb') as f_est:
        modelo_reg_est = pickle.load(f_est)

with open('./df.pkl', 'rb') as f_df:
        df = pickle.load(f_df)

a4_dims = (11.7, 8.27)
#fig, ax = plt.subplots(figsize=a4_dims)
#sns.lineplot(x='Year' ,y='temp_avg', data = df, markers=True, dashes=False)

#plt.yticks([10, 15, 20, 25])



