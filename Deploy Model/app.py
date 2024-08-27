import streamlit as st
import os
from keras.models import load_model
from PIL import Image
from predict import classify

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # لإخفاء التحذيرات فقط


# set title
st.title('Teeth Classification')

# set header
st.header('Please upload a teeth image')

# upload file
file = st.file_uploader(' ', type=['jpeg', 'jpg', 'png'])

# load model
model = load_model("trans_teeth3.keras")

# load class name
with open('labels.txt', 'r') as f:
    class_names = [line.strip().split(' ')[0] for line in f if line.strip()]

# display image and classification result
if file is not None:
    # Create three columns (one empty column for spacing)
    col1, spacer, col2 = st.columns([2, 2, 2])

    # Display image in the first column
    image = Image.open(file).convert('RGB')
    col1.image(image, width=400)  

    # classify image and display result in the second column
    class_name, conf_score = classify(image, model, class_names)
    col2.write(f'## {class_name}')
    col2.write(f'### Score: {conf_score * 100:.2f}%')
