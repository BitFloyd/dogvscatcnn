import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from keras.applications import VGG16
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from skimage.io import imread
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import itertools

from tqdm import tqdm

# This function saves and plots a confusion matrix. plt.show() must be called from outside
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          save_fig_name ='confusion_matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_fig_name+'.png',bbox_inches='tight')


# Returns images and labels. The data structure of the images must be: path/class1, path/class2
def get_data(path='path',size=(64,64)):
    # Returns an array of images and labels for training the CNN
    # Return 0 of cat and 1 for dog

    list_images = []
    labels = []

    for j in os.listdir(path):

        if j == '.DS_Store':
            continue

        list_image_names = os.listdir(os.path.join(path,j))

        print ("LOADING "+str(j)+" IMAGES FROM :", os.path.join(path,j))
        for i in tqdm(list_image_names):

            if i == '.DS_Store':
                continue

            if i.split('.')[0] == 'cat':
                labels.append(0)

            elif i.split('.')[0] == 'dog':
                labels.append(1)

            list_images.append(resize(imread(os.path.join(path,j,i)),output_shape=size))


    return np.array(list_images), np.array(labels)


def return_CNN(input_shape=(64,64,3)):

    # create, compile and return a CNN from this function.
    # Feel free to use your own CNN here. Create and return that CNN from here
    # VGG16 is a much bigger model and takes time to fit and predict. use simpler models for speed
    # I am using a different CNN here, just to show how you can use a pretrained CNN easily.
    # Lots more examples available in keras/applications (Check the documentation)

    base_model = VGG16(weights='imagenet',include_top=False,input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(32,activation='relu')(x)
    op_layer = Dense(1,activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=op_layer)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')

    return model


def main():

    print ("GET FULL DATASET")
    # Get FULL Dataset
    images, labels = get_data(os.path.join('dataset','training_set'))

    print ("SHUFFLE DATASET")
    # Shuffle dataset
    images, labels = shuffle(images,labels)

    print ("SPLIT DATASET - 01")
    # Split Dataset into Training and Test Set
    images_train, images_test, labels_train, labels_test = train_test_split(images,labels,test_size=0.20)

    print ("SPLIT DATASET - 02")
    # Split Training set into Training and Validation Set
    images_train, images_val, labels_train, labels_val = train_test_split(images_train,labels_train,test_size=0.20)

    # Create ImageDataGenerator object for training and validation set

    # No need for rescale here since skimage imread returns 0-1 values
    datagen_train = ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)

    # No need for rescale here since skimage imread returns 0-1 values
    datagen_val   = ImageDataGenerator(featurewise_center=True,
                                   featurewise_std_normalization=True,
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)

    # Fit training set and validation set to datagen

    datagen_train.fit(images_train)
    datagen_val.fit(images_val)

    # Get CNN
    model = return_CNN()


    # Get Keras Callbacks
    es = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=10, verbose=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    mcp = ModelCheckpoint(filepath='model_saved.h5',monitor='val_loss',verbose=1,save_best_only=True)

    print ("START MODEL FITTING")
    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen_train.flow(images_train, labels_train, batch_size=32),
                        steps_per_epoch=len(images_train) / 32, epochs=200, callbacks=[es, rlr,mcp],
                        validation_data=datagen_val.flow(images_val,labels_val,batch_size=32),
                        validation_steps=len(images_val)/32)


    # Get accuracy of model on test_data.

    # If you use a sequential model, you can just use predictions_test = model.predict_classes(images_test)
    predictions_test = (model.predict(images_test)>0.5)

    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print ("ACCURACY ON TEST SET SPLIT")
    print (accuracy_score(labels_test,predictions_test)*100.0)
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # Compute confusion matrix
    print ("CONFUSION MATRIX FOR TEST SET SPLIT")
    cnf_matrix = confusion_matrix(y_true=labels_test,y_pred=predictions_test)


    print ("PLOT CONFUSION MATRIX FOR TEST SET SPLIT")
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['cat','dog'],save_fig_name='confusion_matrix_test_set_split')
    plt.show()


    #Remove variables we dont need anymore to make room in the RAM
    del images_test,images_val,images_train,labels_test,labels_val,labels_train,predictions_test,cnf_matrix

    print ("LOAD NEW TEST SET")
    images_new_test, labels_new_test = get_data(os.path.join('dataset','test_set'))

    print ("PREDICT ON NEW TEST SET")
    predictions_new_test = (model.predict(images_new_test) > 0.5)

    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print ("ACCURACY ON NEW TEST SET")
    print (accuracy_score(labels_new_test,predictions_new_test)*100.0)
    print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # Compute confusion matrix
    print ("CONFUSION MATRIX FOR TEST SET")
    cnf_matrix = confusion_matrix(y_true=labels_new_test,y_pred=predictions_new_test)



    print ("PLOT CONFUSION MATRIX FOR TEST SET")
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['cat','dog'],save_fig_name='confusion_matrix_new_test_set')
    plt.show()

if __name__ == "__main__":
    main()