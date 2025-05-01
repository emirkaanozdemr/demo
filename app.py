import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np

model=load_model("model.h5")
st.title("Yaprak Hastalığını Tahmin Et")
img=st.camera_input("Kamera")
def process_image(input_img):
    if input_img.mode == 'RGBA':
        input_img = input_img.convert('RGB')
    input_img=input_img.resize((110,110)) 
    input_img=np.array(input_img)
    input_img=input_img/255.0
    input_img=np.expand_dims(input_img,axis=0)
    return input_img
if st.button("Tahmin Et") and img is not None:
    img=Image.open(img)
    image=process_image(img)
    prediction=model.predict(image)
    predicted_class=np.argmax(prediction)
    class_names = ["Black Rot", "ESCA", "Healthy", "Leaf Blight"]
    st.write(class_names[predicted_class])
