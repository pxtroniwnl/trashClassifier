import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Cargar el modelo
model = YOLO("C:/Users/Alejandro/Desktop/Trash/runs/classify/train4/weights/best.pt")

def make_prediction(frame):
    results = model(frame)
    names = results[0].names
    probs = results[0].probs
    
    if probs.top1 is not None:
        max_index = probs.top1
        confidence = probs.top1conf.item() * 100  # Multiplicar por 100 para obtener el porcentaje
        
        # Crear un diccionario con las probabilidades de cada clase
        probabilities_dict = {names[i]: probs.data[i].item() * 100 for i in range(len(names))}
        
        # Crear un string para mostrar la clase con mayor probabilidad
        result_str = f"**Clase:** {names[max_index]}  \n**Probabilidad:** {confidence:.2f}%\n\n**Probabilidades de cada clase:**\n"
        
        # Añadir las probabilidades de cada clase al string
        for class_name, probability in probabilities_dict.items():
            result_str += f"**{class_name}:** {probability:.2f}%\n"
        
        return result_str
    else:
        return "No se detectó basura."

# Establecer el título y la descripción de la aplicación
st.set_page_config(page_title="Clasificador de Basura", layout="wide")
st.title("Clasificador de Basura")
st.markdown("""Esta aplicación utiliza un modelo de **YOLO** para clasificar residuos a través de la cámara en tiempo real o mediante imágenes cargadas.""")
        
# Opción para subir imágenes
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

# Contenedor principal
camera = cv2.VideoCapture(0)

# Crear contenedores vacíos para la imagen y la predicción
image_box = st.empty()
prediction_box = st.empty()

# Inicializar el tiempo de la última predicción
last_prediction_time = time.time()

# Estado de la clasificación
if 'is_classifying' not in st.session_state:
    st.session_state.is_classifying = False

# Estilo de botón
button_style = """
<style>
.stButton > button {
    background-color: #4CAF50; /* Verde */
    color: white;
    padding: 15px 32px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border: none;
    border-radius: 8px;
    transition: background-color 0.3s;
}
.stButton > button:hover {
    background-color: #45a049; /* Verde más oscuro */
}
</style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# Colocar un botón para iniciar la predicción de la cámara
if st.button("Iniciar Clasificación desde la cámara"):
    st.session_state.is_classifying = True
    st.success("Clasificación iniciada. ¡Mira la cámara!")

# Botón para detener la clasificación
if st.button("Detener Clasificación"):
    st.session_state.is_classifying = False
    st.success("Clasificación detenida.")
    camera.release()

# Mostrar video en tiempo real si está en modo de clasificación
if st.session_state.is_classifying:
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        
        # Convertir la imagen a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Redimensionar la imagen a 500x500 píxeles
        frame_resized = cv2.resize(frame_rgb, (700, 500))
        
        # Mostrar la imagen en el contenedor de la cámara con un tamaño fijo de 500x500
        image_box.image(frame_resized, channels="RGB", use_container_width=False)  # Ajuste a 500x500
        
        # Actualizar la predicción cada 3 segundos
        current_time = time.time()
        if current_time - last_prediction_time >= 3:
            prediction = make_prediction(frame)
            prediction_box.markdown(prediction)
            last_prediction_time = current_time  # Actualizar el tiempo de la última predicción

# Clasificación de la imagen subida
if uploaded_file is not None:
    # Leer la imagen y convertirla a un formato adecuado
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Mostrar la imagen en el contenedor de la imagen
    image_box.image(image, channels="BGR", use_container_width=True)

    # Realizar la predicción en la imagen subida
    prediction = make_prediction(image)
    prediction_box.markdown(prediction)