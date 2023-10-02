import cv2 # Image manipulation
import matplotlib.pyplot as plt # Image inputing
import numpy as np # Image reshaping and retruning max probability.
import streamlit as st # Web app API
import tensorflow as tf # Main DL library

# Model, import from a .model folder.
model = tf.keras.models.load_model('../tf_CNN_v2.model')

# Title
st.title('Digit Recognizer')

st.subheader('Input Image')
# Input image as type png jpg or jpeg.
input_image = st.file_uploader('Upload an image of a digit', type=['png', 'jpg', 'jpeg'])


def recognize_digit(img):
    '''
    :param img: Image from the user, any size and format support are png, jpg, jpeg.
    :return: tuple: index of max probabilities and vectorized image.
    '''
    vectorized_img = plt.imread(img)
    gray_img = cv2.cvtColor(vectorized_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.resize(gray_img, (28, 28))
    gray_img = gray_img.reshape(1, 28, 28, 1)
    probabilities = tf.nn.softmax(model.predict([gray_img]))
    return np.argmax(probabilities), vectorized_img


output = []
if st.button('Recognize'): # If this button is clicked:
    if input_image is not None: # If there is an input image:
        output = recognize_digit(input_image) #Save the output of the "recognize_digit" function.

        col1, col2 = st.columns(2) #create two columns

        with col1: # Right col for the actual image
            st.subheader("Actual Image")
            st.image(output[1])

        with col2: # Left col for the predixted digit
            st.subheader("Predicted Digit")
            st.header(f":green[{output[0]}]")
