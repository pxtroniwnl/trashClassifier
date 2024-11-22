import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Cargar el modelo
model = YOLO("runs/classify/train4/weights/best.pt")

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
st.markdown("""Esta aplicación utiliza un modelo de **YOLO** para clasificar residuos mediante imágenes cargadas.""")
        
# Opción para subir imágenes
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

# Contenedor para la imagen y la predicción
image_box = st.empty()
prediction_box = st.empty()

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