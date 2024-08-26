import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # لإخفاء التحذيرات فقط

from keras.models import load_model
from PIL import Image
from predict import classify

# set title
st.title('Teeth Classification')

# set header
st.header('Please upload a teeth image')

# upload file
file = st.file_uploader('Upload your image here', type=['jpeg', 'jpg', 'png'])

# load model
model = load_model("trans_teeth3.keras")

# load class name
with open('labels.txt', 'r') as f:
    class_names = [line.strip().split(' ')[0] for line in f if line.strip()]

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write(f'## {class_name}')
    st.write(f'### Score: {conf_score * 100:.2f}%')
