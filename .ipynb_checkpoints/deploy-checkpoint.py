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

with modeltraining:
    
    def final_model(train_images, train_y, test_images, test_y, val_images, val_y):
        st.header('Model Training and Testing')
        st.markdown('This section where we will be training, tuning and testing or convolutional neural network model.')

        epochs = st.selectbox('Choose how many epochs to run. (Each Epoch added adds to training time!)', options=[1,2,3,4,5,6,7,8,9,10], index=0)
        step_per_epoch =  st.selectbox('Choose the amount of steps for each epoch', options=[1,2,3,4,5,6,7,8,9,10], index=0)
        np.random.seed(123)
        model = models.Sequential()
        model = Sequential()
        model.add(layers.Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(layers.Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(layers.Dropout(0.1))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(layers.Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(layers.Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(layers.Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(layers.Flatten())
        model.add(layers.Dense(units = 128 , activation = 'relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(units = 1 , activation = 'sigmoid'))
        model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy'])


        model.compile(loss='binary_crossentropy',
                    optimizer="sgd",
                    metrics=['acc'])

        history = model.fit(train_images,
                            train_y,
                            steps_per_epoch=step_per_epoch,
                            epochs=epochs,
                            batch_size=8,
                            validation_data=(val_images, val_y))



        st.markdown(f"\nTraining Score: {model.evaluate(train_images, train_y)}")
        st.markdown(f"\nTest Score: {model.evaluate(test_images, test_y)}")
    
        return history

    
    st.write(final_model(train_images, train_y, test_images, test_y, val_images, val_y))
