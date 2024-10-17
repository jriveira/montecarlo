import random
import numpy as np
import numpy_financial as npf
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt

class SimuladorVan():
    '''
    Simulación de VAN (NPV) por método de Montecarlo
    Variables Aleatorias para el modelo:
        -Ingresos: Disribución Normal
        -Egresos: Distribución Normal
        -Inversiones: Distribución Pareto o Lasso II
        -Tasa: Distribución Logística

    '''

    def load_data(self, df):
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

        return (in_m, 
                in_stdev, 
                out_m, 
                out_stdev, 
                inv, 
                inv_m, 
                inv_stdev,
                r_m,
                r_stdev,
                iterations)

    def npv_simulator(in_m, 
                    in_stdev,
                    out_m, 
                    out_stdev,
                    inv,
                    inv_m, 
                    inv_stdev,
                    r_m, 
                    r_stdev,
                    iterations=1000):
        ''' Esta función permite generar una simulación del VAN aplicando el método de montecarlo, contemplando las variables
        aleatorias de Ingresos, Egresos, inversiones y tasa de descuento como distribuciones normales.
        
        Args:
            in_m (float): Valor medio estimado para el vector de ingresos.
            in_stdev (float): Desviación estándar estimada para los ingresos.
            out_m (float): Valor medio estimado para el vector de egresos/gastos.
            out_stdev (float): Desviación estándar estimada para los egresos.
            inv_m (float): Valor medio estimado para el vector de inversiones.
            inv_stdev (float): Desviación estándar estimada para las inversiones.
            r_m (float) : Valor medio estimado para la tasa de descuento.
            r_stdev (float) : Desviación estándar estimada para la tasa de descuento.
            iterations (int): Cantidad de muestras a generar en forma aleatoria para realizar las simulaciones.
            
        Returns:
            npv (array): Vector de valores para el VAN (NPV) simulados a partir de las ditribuciones normales de ingresos, egresos, 
            inversiones y tasa de descuento.
        
        Raises:
            None
        
        Notes:
            Sólo se considera en esta funciópn la simulación de las variables mediante una generación de valoresa a partir de
            distribución normal.
            
        '''
        npv=[]
        for i in range(iterations):
            inFlow = np.random.normal(in_m, in_stdev, iterations)
            outFlow = np.random.normal(out_m, out_stdev, iterations)
            invest=np.random.pareto(inv_m,size=iterations)
            #rate = np.random.logistic(loc=r_m,scale=r_stdev, size=iterations)
        
            cashFlow = -invest + inFlow - outFlow
            npv.append(npf.npv(rate=r_m, values=cashFlow))
        npv=np.array(npv)
        
        #Depura los valores sin definir para evitar errores
        mask = np.isfinite(npv)
        npv=npv[mask]
        return npv
    
    def plot_pdf(npv):
        # Crear un estimador de la densidad con la función KDE
        kde = gaussian_kde(npv)
        
        # Generar un rango de valores para graficar la densidad
        x_range = np.linspace(min(npv), max(npv), len(npv))
        
        # Evaluar la densidad para los valores generados
        pdf_values = kde(x_range)
        
        # Graficar la función de densidad
        fig, ax = plt.subplots()
        ax.plot(x_range, pdf_values, label='Densidad de Probabilidad')
        ax.fill_between(x_range, pdf_values, alpha=0.3)
        ax.set_xlabel('Valor de NPV')
        ax.set_ylabel('Densidad')
        ax.set_title('Función de Densidad de Probabilidad')
        ax.legend()
        ax.grid(True)

        return fig

    def plot_cdf(npv):
        # Ordenar los valores de npv
        sorted_npv = np.sort(npv)
        
        # Crear el vector de probabilidades acumuladas
        cdf = np.arange(1, len(npv) + 1) / len(npv)


        # Calcular el valor de la CDF en npv=0
        cdf_value_at_zero = cdf[np.searchsorted(sorted_npv, 0, side='right') - 1]


        # Graficar la CDF
        fig, ax = plt.subplots()
        ax.plot(sorted_npv, cdf, label='Función de Probabilidad Acumulada (CDF)', color='blue')
        ax.fill_between(sorted_npv, cdf, alpha=0.3, color='blue')
        ax.set_xlabel('Valor de NPV')
        ax.set_ylabel('Probabilidad acumulada')
        ax.set_title('Función de Probabilidad Acumulada (CDF)')
        ax.legend()
        ax.grid(True)


        # Agregar una línea vertical de referencia en npv=0
        plt.axvline(x=0, color='red', linestyle='--', label='Línea de referencia (npv=0)')

        # Mostrar el valor de la CDF en npv=0 como texto
        plt.text(0, cdf_value_at_zero, f'P(VAN<=0)={cdf_value_at_zero:.2f}', color='black', 
            ha='left', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        return fig