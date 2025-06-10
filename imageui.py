import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


model = tf.keras.models.load_model('CatandDog.h5')

st.title('Cat and Dog classifier')

upload = st.file_uploader('Upload an Image', type=['jpg', 'png'])

img_size = 100

if upload:
    image =  Image.open(upload)
    st.image(image, caption='Uploaded image')

    image = image.resize((img_size, img_size))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis = 0)


    prediction = model.predict(image)
    class_idx = np.argmax(prediction)
    st.write(f'Predicted class is {class_idx}')
    


