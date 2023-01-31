import streamlit as st
import pandas as pd
import pickle

# Load model
with open("svml_model.pkl", "rb") as f:
    model = pickle.load(f)

st.write("""
# Iris Flower Prediction App
Aplikasi ini memprediksi spesies bunga Iris berdasarkan panjang sepal, lebar sepal, panjang petal, dan lebar petal.
""")
st.write("Adi Firmansyah - 2019230083")

# User input
sepal_length = st.slider("Sepal length", 4.3, 7.9, 5.4)
sepal_width = st.slider("Sepal width", 2.0, 4.4, 3.4)
petal_length = st.slider("Petal length", 1.0, 6.9, 1.3)
petal_width = st.slider("Petal width", 0.1, 2.5, 0.2)

# Prediction
if st.button("Predict"):
    result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    species = result[0]
    st.write("The species of this flower is:", species)
    
    # Show image
    if species == "Iris-setosa":
        st.image("iris_setosa.jpg")
    elif species == "Iris-versicolor":
        st.image("iris_versicolor.jpg")
    elif species == "Iris-virginica":
        st.image("iris_virginica.jpg")


