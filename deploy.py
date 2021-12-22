import streamlit as st
import numpy as np 
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import plot_confusion_matrix
from tqdm import tqdm 
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import array_to_img
from code.preparation import img_data_gen
from keras.models import Sequential
from tensorflow.keras import layers, models
from glob import glob
from PIL import Image
import joblib
import pickle
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from tensorflow.keras import preprocessing
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

    val_images, val_labels = next(val_gen)
    test_images, test_labels = next(test_gen)
    train_images, train_labels = next(train_gen)

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
import seaborn as sns

with dataset:
    
    st.header("Data Understanding")
    
    st.markdown("Our data comes from Mendeley Data, it contains a few thousand images Chest X-Ray described and analyzed in 'Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning'. The images are split into a training set and a testing set of independent patients. Images are labeled as (disease)-(randomized patient ID)-(image number by this patient). We are going to be classifying whether the images fall into the two classes either 'NORMAL' or 'PNEUMONIA'.")

    test_images, train_images, val_images, val_labels, test_labels, train_labels = Load_Xray_Images___Will_take_some_time_but_should_only_need_to_run_once___ETA_1_to_5_MIN()
    
    train_y = np.reshape(train_labels[:,0], (5217,1))
    test_y = np.reshape(test_labels[:,0], (624,1))
    val_y = np.reshape(val_labels[:,0], (16,1))


    
with features:
        
    st.header('Data Preview')
    
#     sel_col, disp_col = st.columns(2)
    st.markdown('A random selection of images from the dataset.')
    
    train_norm = 'data/chest_xray/train/NORMAL'
    train_pneu = 'data/chest_xray/train/PNEUMONIA'
    
    train_normal_images = glob(train_norm + '/*')
    train_pneumonia_images = glob(train_pneu + '/*')

    fig = plt.figure(figsize=(10, 8), dpi=80)
    init_subplot = 230
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        if i < 4:
            img = Image.open(np.random.choice(train_normal_images)).resize((244, 244))
            plt.title("Normal Image")
        else:
            img = Image.open(np.random.choice(train_pneumonia_images)).resize((244, 244))
            plt.title("Pneumonia Image")
        img = np.asarray(img)
        plt.axis('off')
        plt.imshow(img)
    st.pyplot(fig)
    
    fig  = plt.figure(figsize=(5, 5))
    mask = []
    for i in train_labels:
        if(i[1] == 1):
            mask.append("Pneumonia")
        else:
            mask.append("Normal")
    st.sidebar.subheader('% of Images in Each Class.')        
    st.sidebar.markdown(pd.DataFrame(mask).value_counts(normalize=True))

    sns.countplot(mask)
    st.sidebar.pyplot(fig)
    
fig = plt.figure()

def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)


def predict(image):
    classifier_model = "final_cnn_model.h5"
    IMAGE_SHAPE = (224, 224,3)
    model = load_model(classifier_model, compile=False)
#     test_image = image.resize((224,224))
#     test_image = preprocessing.image.img_to_array(test_image)
#     test_image = test_image /255
#     test_image = np.expand_dims(test_image, axis=0)
    class_names = [
          'Normal',
          'Pneumonia']
    predictions = model.predict(image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
          'Normal': 0,
          'Pneumonia': 0
}

    
    result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence." 
    return result


if __name__ == "__main__":
    main()