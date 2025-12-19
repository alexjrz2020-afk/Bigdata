
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.sql.functions import col
import zipfile
import os

# Iniciar Spark
spark = SparkSession.builder.appName("ProyectoBigData").getOrCreate()

# Ruta del archivo comprimido
model_path = "/content/modelo_logistico_spark.zip"
extract_folder = "/content/modelo_logistico_spark/"

# Verificar si la carpeta del modelo ya existe
if not os.path.exists(extract_folder):
    with zipfile.ZipFile(model_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
        print(f"Modelo descomprimido en: {extract_folder}")
else:
    print(f"El modelo ya está descomprimido en: {extract_folder}")

# Intentar cargar el modelo
try:
    lr_model = LogisticRegressionModel.load(extract_folder)
    print("Modelo cargado correctamente")
    print(f"Tipo de modelo: {type(lr_model)}")
except Exception as e:
    print(f"Hubo un problema al cargar el modelo: {e}")

# Descripción del modelo en Streamlit
st.title("Aplicación de Predicción con Regresión Logística")
st.write("""
Este modelo predice la probabilidad de que una muestra pertenezca a una clase específica.
Las predicciones se basan en los valores de las variables D01 a D12, que representan diferentes características de entrada.
""")

# Visualización de métricas de desempeño
st.header("Métricas de Evaluación del Modelo")
st.write(f"**Accuracy:** 0.7102")
st.write(f"**Precision:** 0.7307")
st.write(f"**Recall:** 0.7102")
st.write(f"**F1-Score:** 0.7050")

# Ingreso de valores para la predicción (D01 a D12) usando Sliders
st.header("Ingrese los valores de D01 a D12 para hacer una predicción")

# Usamos deslizadores (sliders) para permitir al usuario ingresar valores
input_data = []
for i in range(1, 13):
    value = st.slider(f"Valor D{i:02d}", min_value=0.0, max_value=100.0, step=0.1)
    input_data.append(value)

# Realizar predicción si el usuario ingresa los datos
if input_data:
    # Convertir los valores ingresados en un DataFrame de Spark
    cols = [f"D{str(i).zfill(2)}" for i in range(1, 13)]  # D01, D02, ..., D12
    user_data = spark.createDataFrame([input_data], cols)

    # Realizar el ensamblaje de las características (VectorAssembler)
    assembler = VectorAssembler(inputCols=cols, outputCol="features")
    user_data = assembler.transform(user_data)

    # Realizar la predicción
    user_prediction = lr_model.transform(user_data)
    prediction = user_prediction.select('prediction').first()[0]

    # Mostrar el resultado de la predicción
    st.write(f"**Predicción:** {prediction}")
    st.write("Este valor indica la clase predicha por el modelo basado en las características ingresadas.")

# Agregar más interactividad si es necesario (como gráficos, más información, etc.)
st.sidebar.header("Más información")
st.sidebar.write("""
    Puedes experimentar con los valores de las variables D01 a D12 para observar cómo cambian las predicciones del modelo.
    Además, puedes consultar los detalles sobre el rendimiento del modelo y el análisis.
""")
