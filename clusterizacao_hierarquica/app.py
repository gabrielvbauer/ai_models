import streamlit as st
import pandas as pd

@st.cache_data
def carregar_dados():
  return pd.read_csv('./datasets/cluster_laptops.csv')

df = carregar_dados()

st.sidebar.header('Filtros')

model = st.sidebar.selectbox('Selecionar Modelo', df['model'].unique())

df_laptops_model = df[df['model'] == model]

df_laptops_final = df[df['cluster'] == df_laptops_model.iloc[0]['cluster']]

st.write('Recomendações de modelos')
st.table(df_laptops_final)