import streamlit as st
import numpy as np 
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import plot_confusion_matrix
from tqdm import tqdm 
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import array_to_img
np.random.seed(123)


header = st.container()
dataset = st.container()
features = st.container()
modeltraining = st.container()

@st.cache
def Load_Xray_Images___Will_take_some_time_but_should_only_need_to_run_once___ETA_1_to_5_MIN():

    train_path = 'data/chest_xray/train/'
    test_path = 'data/chest_xray/test/'
    val_path = 'data/chest_xray/val'
    np.random.seed(123)
    train_gen, test_gen, val_gen = img_data_gen(train_path, test_path, val_path)

    val_images, val_labels = tqdm(next(val_gen))
    test_images, test_labels = tqdm(next(test_gen))
    train_images, train_labels = tqdm(next(train_gen))

    return test_images, train_images, val_images, val_labels, test_labels, train_labels

with header:
    st.title('Welcome to the Pneumonia Classification Project')
    st.markdown(
    """
    <style>
   .sidebar .sidebar-content {
        background: #FFED91;
    }
    </style>
    """,
    unsafe_allow_html=True
)
    st.markdown("The number of Pneumonia cases increased in the last few years with 1.5 million ER visits and 2 million deaths indicating around 16% increase. The goal of our project is to build a image classification model that can correctly identify between x-rays of infected and healthy lungs so we can lower these numbers. It's important that our model has high accuracy.")

import code.preparation as prep
import code.visualization as viz    

with dataset:
    
    st.header("Data Understanding")
    
    st.markdown("Our data comes from Mendeley Data, it contains a few thousand images Chest X-Ray described and analyzed in 'Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning'. The images are split into a training set and a testing set of independent patients. Images are labeled as (disease)-(randomized patient ID)-(image number by this patient). We are going to be classifying whether the images fall into the two classes either 'NORMAL' or 'PNEUMONIA'.")

    test_images, train_images, val_images, val_labels, test_labels, train_labels = Load_Xray_Images___Will_take_some_time_but_should_only_need_to_run_once___ETA_1_to_5_MIN()
    
    train_y = np.reshape(train_labels[:,0], (5217,1))
    test_y = np.reshape(test_labels[:,0], (624,1))
    val_y = np.reshape(val_labels[:,0], (16,1))


    
with features:
    st.header('Data Preview')

    fig  = plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(array_to_img(train_images[i]))
        if train_y[i] == 0:
            plt.title("Pneumonia")
        else:
            plt.title("Normal")
        plt.axis("off")

    st.pyplot(fig)
  