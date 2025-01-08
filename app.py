import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('my_model.keras')

# Define class names
class_names = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

# Streamlit app
st.title("Diabetes Detection from Fundus Images")
st.write("Upload a fundus image to predict the severity of diabetic retinopathy.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)

    # Predict the label
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]

    # Display the image and prediction
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=False, width=300)
    st.write(f"**Predicted Class:** {predicted_class}")

    # Plot the image with the predicted label
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(predicted_class)
    ax.axis('off')
    st.pyplot(fig)
