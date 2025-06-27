import pandas as pd
import numpy as np
from IPython.display import display

# Visualizaciones
import matplotlib.pyplot as plt    
import seaborn as sns

def analisis_rapido(df, n=5):
     
     """Función que porporciona un análisis rápìdo de un DataFrame.

     Args:
         df (DataFrame): DataFrame a analizar.
         n (int, optional): Númeo de filas, por defecto 5.
     """
   
     print(f"Las {n} primeras filas son:\n")
     display(df.head(n))
     print("___________________________")
     print("Información básica del DataFrame:\n")
     display(df.info())
     print("___________________________")
     print(f"El número de duplicados es {df.duplicated().sum()} ")
     print("___________________________")
     print("Porcentaje de nulos:\n")
     display(df.isna().mean().round(4) * 100)



def eda(df, n=2):
    """Función que proporciona un EDA rápido

    Args:
        df (DataFrame): DataFrame sobre el que queramos hacer el EDA
        n (int, optional): Número de decimales que queramos para redondear, por defecto 2.
    """
     
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include=['object','category']).columns
     
    print("Variables numéricas:\n\n", num_cols)
    print('___________________________')
    print("Variables categóricas:\n\n", cat_cols)
    print('___________________________')
     
    print("\n\nVeamos las estadísticas básicas:\n")
    display(df.describe().T.round(n))
    display(df.describe(include=['object','category']).T.round(n))
     
    for col in cat_cols:
          print(f"\n--------------ESTAMOS ANALIZANDO LA COLUMNA: '{col}'----------------------\n")
          print(f"Valores únicos: {df[col].unique()}\n")
          print(f"Frecuencia de los valores únicos de las categorías:")
          display(df[col].value_counts())

    print("Countplot de las columans categóricas")
    for col in cat_cols:
        if df[col].nunique() > 200:
            print(f"Columna {col} tiene demasiadas categorías {df[col].nunique()}\n\n")
            continue
        
        num_categories = df[col].nunique()
        width = max(7,num_categories * 0.5)
        height = 3
        
        plt.figure(figsize=(width,height))
        sns.countplot(x=df[col], order=df[col].value_counts().index)
        
        plt.title(f"Gráfico de barras de: {col}")
        plt.xlabel(col)
        plt.xticks(rotation=90)
        
        plt.show()
    
    print("Vamos con los histogramas de las columnas numéricas\n")    
    for col in num_cols:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col], bins=30, edgecolor='black')
     
        plt.title(f"Distribución de  {col}")
        plt.xlabel(col)
     
        plt.show()
    
    print("Vayamos con los boxplot")    
    for col in num_cols:
        plt.figure(figsize=(10, 1))
        sns.boxplot(x=df[col])
     
        plt.title(f"Distribución de  {col}")
        plt.xlabel(col)
        plt.ylabel("Frecuencia")
     
        plt.show()
     
     
     
def matriz_correlacion(df):
     """Funcion que muestra la matriz de correlacion

     Args:
        df (DataFrame): _DataFrame a analizar
    """

     #Crear la figura
     plt.figure(figsize=(len(df.columns), len(df.columns)))

     #Crear una máscara para mostrar sólo la parte triangular
     mask = np.triu(np.ones_like(df.corr(numeric_only=True), dtype=bool))

     #Graficar el mapa de calor
     sns.heatmap(data=df.corr(numeric_only=True),
               annot=True,
               annot_kws={"size": 12, "color": "black"},
               mask=mask,
               fmt='.2f',
               vmin=-1,
               vmax=1,
               cmap='cool',
               linewidths=0.5,
               linecolor='black')

     plt.title('Matriz de correlación', fontsize= 14)
     plt.xticks(rotation=45)
     plt.yticks(rotation=0)
     plt.tight_layout()
     plt.show()