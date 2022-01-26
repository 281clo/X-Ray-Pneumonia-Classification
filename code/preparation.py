
import numpy as np  
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.models import Sequential
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import f1_score


class_weights = {0: 1.9444858420268256, 1: 0.6730719628578798}


def img_data_gen(train_path, test_path, val_path):
    
    train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(train_path,
                                                                                  batch_size=5217,
                                                                                  target_size=(256,256),
                                                                                  color_mode = 'grayscale')

    # get all the data in the directory chest_xray/test (624 images), and reshape them
    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(test_path,
                                                                            batch_size = 624,
                                                                            target_size=(256,256),
                                                                            color_mode = 'grayscale') 

    # get all the data in the directory split/validation (16 images), and reshape them
    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(val_path,
                                                                           batch_size = 16,
                                                                           target_size=(256,256),
                                                                           color_mode = 'grayscale')
    return train_gen, test_gen, val_gen

def img_data_gen_batched(train_path, test_path, val_path):
    
    train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(train_path,
                                                                       batch_size=16,
                                                                       target_size=(256,256),
                                                                       color_mode = 'grayscale')

    # get all the data in the directory chest_xray/test (624 images), and reshape them
    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(test_path,
                                                                      batch_size = 16,
                                                                      target_size=(256,256),
                                                                      color_mode = 'grayscale') 

    # get all the data in the directory split/validation (16 images), and reshape them
    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(val_path,
                                                                     batch_size = 4,
                                                                     target_size=(256,256),
                                                                     color_mode = 'grayscale')
    return train_gen, test_gen, val_gen

def create_sets(train_gen, test_gen, val_gen):

    train_images, train_labels = next(train_gen)
    test_images, test_labels = next(test_gen)
    val_images, val_labels = next(val_gen)

    return (train_images, train_labels), (test_images, test_labels), (val_images, val_labels)


def logistic_regression_l2(train_img, train_y, test_img, test_y):
    log = LogisticRegression(penalty='l2')
    log.fit(train_img, train_y)
    print(f'Accuracy test score of logistic regression model with L2 regularization: {log.score(test_img, test_y)}')

    return log


def random_forest(train_img, train_y, test_img, test_y):
    RF_model = RandomForestClassifier(criterion= 'entropy', max_depth= 15, min_samples_split= 5, 
                               n_estimators= 700, random_state=777,max_features='log2')
    RF_model.fit(train_img, train_y)
    print(f'Cross validation score for Random Forest: {cross_val_score(RF_model, test_img, test_y, cv=5)}')
    print(f'Accuracy test score of random forest model: {RF_model.score(test_img, test_y)}')

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
                metrics=[tf.keras.metrics.PrecisionAtRecall(0.5)])

    history = model.fit(train_images,
                        train_y,
                        steps_per_epoch=20,
                        epochs=20,
                        batch_size=16,
                        validation_data=(val_images, val_y))

    print(f"Training Score of first convolution neural network: {model.evaluate(train_images, train_y)[1]}")
    print(f"Test Score of first convolution neural network: {model.evaluate(test_images, test_y)[1]}")

    return history

def final_model(train_gen, val_gen, test_gen):

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), padding = 'same', input_shape=(256, 256, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2,2) , padding = 'valid'))

    model.add(layers.RandomTranslation(height_factor=0.1, width_factor=0.1))
    model.add(layers.RandomFlip())

    model.add(layers.Conv2D(64 , (3,3) , padding = 'valid'))
    model.add(layers.Dropout(0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2,2) , padding = 'valid'))
    
    model.add(layers.Conv2D(128 , (3,3) , padding = 'valid'))
    model.add(layers.Dropout(0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2,2) , padding = 'valid'))

    model.add(layers.Conv2D(256 , (3,3) , padding = 'valid'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D((2,2) , padding = 'valid'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.15))
    model.add(layers.Dense(2, activation='sigmoid'))


    model.build([None, 256, 256, 3])
    model.summary()

    model.compile(loss='binary_crossentropy', 
                  metrics=[tf.keras.metrics.PrecisionAtRecall(0.5)],
                  optimizer=SGD(.0001, .9))
    
    filepath = 'Models/model.epoch{epoch:02d}-loss{val_loss:.2f}.h5'
    checkpoint = ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=False,
                             mode='min')

    history = model.fit(train_gen,
                        steps_per_epoch=train_gen.n // train_gen.batch_size,
                        epochs=25,
                        validation_data=val_gen,
                        validation_steps=val_gen.n // val_gen.batch_size,
                        class_weight=class_weights,
                        callbacks=checkpoint)

    
    return history

