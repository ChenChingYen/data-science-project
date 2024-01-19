import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="About Model",
    page_icon="📖",
    layout="wide"
)

st.title("✈︎ About Model")
st.subheader("The architecture of the two tower deep neural network recommendation model using tensorflow keras.")

dnn_image_path = "E:/data_science_project/pages/visualizations/dnn_model_architecture.png"
st.image(Image.open(dnn_image_path), output_format='png', width=512)

st.write("This Deep Neural Network Recommendation Model is built based on the architecture of the Tensorflow Two-towers architectures for deep retrieval used by YouTube. The user features are passed into the user tower and the product features are passed into the product tower. After iterating through the dense layer of each tower. The output of the model is the predicted value of the 'rating' score a certain user will provide to that product.")