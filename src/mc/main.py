import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from montecarlo_m import SimuladorVan

st.title("Simulador de VAN")
st.sidebar.markdown("## SIMULADOR VAN")

archivo = st.file_uploader("Subir archivo csv con datos ",
                           type="csv")
if archivo is not None:
    df = pd.read_csv(archivo)

    st.subheader("Visualización de datos")
    st.table(df)

    st.subheader("Resumen de datos")
    st.table(df.describe())

    #in_m,in_stdev,out_m,out_stdev,inv,inv_m,inv_stdev,r_m,r_stdev,iterations = SimuladorVan.load_data(df=df)
    
    in_m=df.Ingresos.mean() 
    in_stdev=df.Ingresos.std()
    out_m=df.Costos.mean()
    out_stdev=df.Costos.std()
    inv=df.Inversiones
    inv_m=df.Inversiones.mean() 
    inv_stdev=df.Inversiones.std()
    r_m=df.Tasa.mean()
    r_stdev=df.Tasa.std()
    iterations = 1000

    van=SimuladorVan.npv_simulator(in_m=in_m, 
                                 in_stdev=in_stdev,
                                 out_m=out_m, 
                                 out_stdev=out_stdev,
                                 inv=inv,
                                 inv_m=inv_m, 
                                 inv_stdev=inv_stdev,
                                 r_m=r_m, 
                                 r_stdev=r_stdev,
                                 iterations=iterations)
    
    st.subheader("Distribución de probabilidad de VAN")


    st.pyplot(SimuladorVan.plot_pdf(van))


    st.subheader("Probabilidad acumulada de VAN")
    

    st.pyplot(SimuladorVan.plot_cdf(van))

else:
    st.write("Esperando la carga del archivo...")







