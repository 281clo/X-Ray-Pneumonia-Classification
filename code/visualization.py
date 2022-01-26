import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from keras.preprocessing.image import array_to_img
import numpy as np

def class_inbalance(train_labels):

    mask = []
    for i in train_labels:
        if(i[1] == 1):
            mask.append("Pneumonia")
        else:
            mask.append("Normal")
    print(pd.DataFrame(mask).value_counts(normalize=True))

    sns.set_style('darkgrid')
    return sns.countplot(mask)

def preview_img(train_images, train_y):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(array_to_img(train_images[i]))
        if train_y[i] == 0:
            plt.title("Pneumonia")
        else:
            plt.title("Normal")
        plt.axis("off")

    return plt.show()

def view_img_mean(train_images, train_y):    
    pneum = []
    norm = []
    for i in range(5217):
        if train_y[i][0] == 0:
            pneum.append(train_images[i])
        elif train_y[i][0] == 1:
            norm.append(train_images[i])
    mean_norm = np.mean(norm, axis=0)
    mean_pneum = np.mean(pneum, axis=0)
    fig = plt.subplots(figsize=(20,20))
    plt.subplot(1,3,1)
    plt.imshow(array_to_img(mean_norm))
    plt.title('Normal Average')
    plt.subplot(1,3,2)
    plt.imshow(array_to_img(mean_pneum))
    plt.title('Pnumonia Average')
    
    contrast_mean = mean_norm - mean_pneum
    plt.subplot(1,3,3)
    plt.imshow(contrast_mean, cmap='bwr')
    plt.title(f'Difference Between Normal & Pneumonia Average')
    plt.axis('off')
    plt.show()

def accuracy_and_losses(history):
    acc = history.history['precision_at_recall']
    val_acc = history.history['val_precision_at_recall']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation Precision at Recall')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    return plt.show()


def accuracy_and_losses2(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    return plt.show()