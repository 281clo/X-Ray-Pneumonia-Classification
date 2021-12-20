from sklearn.linear_model import LogisticRegression
import numpy as np  
from tqdm import tqdm 
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator




def img_data_gen(train_path, test_path, val_path):
    # get all the data in the directory split/train (5216 images), and reshape them
    train_generator = ImageDataGenerator(rescale=1./255, shear_range = 0.2, zoom_range = 0.2).flow_from_directory(
            train_path,
            batch_size=5217)

    # get all the data in the directory chest_xray/test (624 images), and reshape them
    test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
            test_path,
            batch_size = 624) 

    # get all the data in the directory split/validation (16 images), and reshape them
    val_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
            val_path,
            batch_size = 16)

    

    return train_generator, test_generator, val_generator

def create_sets(train_gen, test_gen, val_gen):

    train_images, train_labels = tqdm(next(train_gen))
    test_images, test_labels = tqdm(next(test_gen))
    val_images, val_labels = tqdm(next(val_gen))

    return (train_images, train_labels), (test_images, test_labels), (val_images, val_labels)

def data_aug():
    dataAug = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 45,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


    return dataAug

def logistic_regression_l2(train_img, train_y, test_img, test_y):
    log = LogisticRegression(penalty='l2')
    log.fit(train_img, train_y)
    print(f'Test score of logistic regression model with L2 regularization: {log.score(test_img, test_y)}')

    return log


def random_forest(train_img, train_y, test_img, test_y):
    RF_model = RandomForestClassifier(criterion= 'entropy', max_depth= 15, min_samples_split= 5, 
                               n_estimators= 700, random_state=777,max_features='log2')
    RF_model.fit(train_img, train_y)
    print(f'Cross validation score for Random Forest: {cross_val_score(RF_model, test_img, test_y, cv=5)}')
    print(f'Test score of random forest model: {RF_model.score(test_img, test_y)}')

    return RF_model


def first_cnn(train_images, train_y, val_images, val_y, test_images, test_y):
    np.random.seed(123)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l1_l2(.005))),
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer="sgd",
                metrics=['acc'])

    history = model.fit(train_images,
                        train_y,
                        steps_per_epoch=20,
                        epochs=20,
                        batch_size=16,
                        validation_data=(val_images, val_y))



    print(f"Training Score of first convolution neural network: {model.evaluate(train_images, train_y)}")
    print(f"Test Score of first convolution neural network: {model.evaluate(test_images, test_y)}")

    return history

def final_model(train_images, train_y, test_images, test_y, val_images, val_y):
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
                        steps_per_epoch=50,
                        epochs=20,
                        batch_size=8,
                        validation_data=(val_images, val_y))



    print(f"\nTraining Score: {model.evaluate(train_images, train_y)}")
    print(f"\nTest Score: {model.evaluate(test_images, test_y)}")
    
    return history

