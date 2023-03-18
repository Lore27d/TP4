import streamlit as st
import pandas as pd
#import seaborn as sns
#from matplotlib import pyplot as plt
import pickle
from datetime import date, time, datetime

st.title("Predicción de Temperatura")
st.write("Escribiendo...")


"""
dataframe = pd.DataFrame(np.random.randn(10, 20),columns=[f'col {i}' for i in range(20)])

st.dataframe(dataframe.style.highlight_max(axis=0))
st.table(dataframe)

map_data = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],columns=['lat', 'lon'])
st.map(map_data)

"""

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

df

fecha = input("Ingresar mes y fecha a predecir: ")

def prediccion_fecha(fecha_pred):
    
    df_pred = pd.DataFrame()
    df_pred.set_index = df.index
    df_pred['Year'] = df.index.year
    df_pred['Month'] = df.index.month
    df_pred

    fecha = datetime.strptime(fecha_pred, '%Y/%m')

    años = []
    año = 2022
    for i in range(0,(fecha.year-año)+1):
        años.append(año)
        año += 1
    
    for año in años:
        if(año != años[-1]):
            i = 12
        elif(año == años[-1]):
            i= fecha.month
        for i in range (0,i):
            df_pred.loc[df_pred.shape[0]] = [año,i+1]

    dummies_mes_pred = pd.get_dummies(df_pred["Month"], drop_first=True)
    dummies_pred=pd.DataFrame(dummies_mes_pred)
    dummies_pred=dummies_pred.rename(columns={2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',8:'aug',9:'sep',10:'oct',11:'nov',12:'dec'})

    dummies_pred.index = df_pred.index
    df_pred=pd.merge(df_pred, dummies_pred, left_index=True, right_index=True)

    df_pred['timeIndex']=df_pred.index
    df_pred['Fecha'] = str(df_pred['Year']) + '-' + str(df_pred['Month'])

    for i in range (0, df_pred.shape[0]):
        df_pred['Fecha'][i] = datetime.strptime(str(df_pred['Year'][i]) +'-'+ str(df_pred['Month'][i]), "%Y-%m")   

    pred_reg = modelo_reg_est.predict(df_pred[['timeIndex','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']])
    pred_arima = modelo_arima.get_prediction(start='1909-01',end=fecha)
    pred_arima = pred_arima.predicted_mean


    predicciones = [ pred_reg[x] + pred_arima[x] for x in range(0,len(pred_reg))] 
    prediccion = pd.DataFrame()
    prediccion['Fecha'] = df_pred['Fecha']

    prediccion['Temp'] = predicciones
    return prediccion

predicción = prediccion_fecha(fecha)

