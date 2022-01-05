import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from keras.preprocessing.image import array_to_img

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


def accuracy_and_losses(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_acc']
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