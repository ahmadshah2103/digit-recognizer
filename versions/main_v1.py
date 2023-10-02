import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

model = tf.keras.models.load_model('../model/tf_CNN_model.model')

st.title('Digit Recognizer')

st.subheader('Input Image')
input_image = st.file_uploader('Upload an image of a digit', type=['png', 'jpg', 'jpeg'])


def recognize_digit(img):
    vectorized_img = plt.imread(img)
    gray_img = cv2.cvtColor(vectorized_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (28, 28))
    gray_img = gray_img.reshape(1, 28, 28, 1)
    probabilities = tf.nn.softmax(model.predict([gray_img]))
    return np.argmax(probabilities), vectorized_img


output = []
if st.button('Recognize'):
    if input_image is not None:
        output = recognize_digit(input_image)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Actual Image")
            st.image(output[1])

        with col2:
            st.subheader("Predicted Digit")
            st.header(f":green[{output[0]}]")
