import numpy as np
import pandas as pd
import random
import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import expand_dims
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, MaxPool2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.inception_v3 import InceptionV3
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import os
import random
import keras
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.models import Model
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from kerastuner.tuners import RandomSearch
import keras_tuner

########## GLOBAL VARIABLES ########
# baseModel=VGG16(input_shape=image_size3d, weights='imagenet', include_top=False)
number_of_classes = 4

#FOLDER OF DATA
#data_dir = 'flowers_recognition//flowers'
data_dir = os.path.join('Corn or Maize Leaf Disease Dataset', 'data')

trainable_tune = True

def import_dataset(number_of_classes, data_dir):
    import os
    folders = os.listdir(data_dir)
    print(folders)
    filenames = []
    categories = []

    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        # HOW MANY IMAGES DO YOU WANT FROM DATASET
        num_images = 0
        for filename in os.listdir(folder_path):
            # if num_images == 500:
            #     break
            filenames.append(os.path.join(folder, filename))
            categories.append(folder)
            num_images += 1

    # Encode the categories as integers
    label_encoder = LabelEncoder()
    categories_encoded = label_encoder.fit_transform(categories)
    categories_encoded = str(categories_encoded)

    df = pd.DataFrame({
        'filename': filenames,
        'category': list(categories)
    })
    return df

# PREPROCESS IMAGES !!
def train_data_gen(train_data, data_dir, image_size2d, batch_size ):
    train_data_gen = ImageDataGenerator(
        rotation_range=15,
        rescale=1. / 255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    trainGeneratorBuild = train_data_gen.flow_from_dataframe(
        train_data,
        directory=data_dir,
        x_col='filename',
        y_col='category',
        target_size=image_size2d,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True
    )
    return trainGeneratorBuild

def validation_gen (valid_data, data_dir, image_size2d, batch_size):
    valid_data_gen = ImageDataGenerator(rescale=1. / 255)
    valGeneratorBuild = valid_data_gen.flow_from_dataframe(
        valid_data,
        directory=data_dir,
        x_col='filename',
        y_col='category',
        target_size=image_size2d,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False  # to alla9a se true
    )
    return valGeneratorBuild

def test_gen (test_data, data_dir, image_size2d, batch_size):
    test_data_gen = ImageDataGenerator(rescale=1. / 255)
    testGeneratorBuild = test_data_gen.flow_from_dataframe(
        test_data,
        directory=data_dir,
        x_col='filename',
        y_col='category',
        target_size=image_size2d,
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=False  # to alla9a se true
    )
    return testGeneratorBuild

def prepare_the_data(number_of_classes, df, data_dir, image_size2d=(224, 224), batch_size=32):
    # for train data
    train_data, test_valid_data = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
    train_data = train_data.reset_index(drop=True)

    train_data, unlabelled = train_test_split(train_data, test_size=0.3, random_state=42, shuffle=True) ###
    test_valid_data = test_valid_data.reset_index(drop=True)

    valid_data, test_data = train_test_split(test_valid_data, test_size=0.5, random_state=42, shuffle=True,
                                             stratify=test_valid_data['category'])
    test_data = test_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)

    # FOR ACTIVE LEARNING UNLABELLED DATA
    unlabelled_df = unlabelled.reset_index(drop=True) ####
    print('train data: ', len(train_data))
    print('test data: ', len(test_data))
    print('val data: ', len(valid_data))
    print('unlabelled data: ', len(unlabelled)) ######

    trainGeneratorBuild = train_data_gen(train_data,data_dir,image_size2d,batch_size)
    valGeneratorBuild = validation_gen(valid_data,data_dir,image_size2d,batch_size)
    testGeneratorBuild = test_gen(valid_data,data_dir,image_size2d,batch_size)
    unlabelledGeneratorBuild = test_gen(unlabelled_df,data_dir,image_size2d,batch_size) ######
    #unlabelledGeneratorBuild = [] ###
    #unlabelled_df = [] ####

    return trainGeneratorBuild, testGeneratorBuild, valGeneratorBuild, unlabelledGeneratorBuild, unlabelled_df, train_data

def which_cnn(cnn, number_of_classes, image_size3d, trainable, tuner):
    if cnn == 1:
        baseModel = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet',
                                                            input_shape=image_size3d, classifier_activation='sigmoid')
        if trainable == False:
            for layer in baseModel.layers:
                layer.trainable = False
        headModel = baseModel.output
        headModel = AveragePooling2D(padding='same')(headModel)
        headModel = Flatten()(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.2)(headModel)
        headModel = Dense(number_of_classes, activation='softmax')(headModel)
        model = Model(inputs=baseModel.input, outputs=headModel)
        print('Model: InceptionV2')
    if cnn == 2:
        baseModel = VGG16(input_shape=image_size3d, weights='imagenet', include_top=False)
        if trainable == False:
            for layer in baseModel.layers[:16]:
                layer.trainable = False
        model = Sequential()
        model.add(baseModel)
        model.add(MaxPool2D((2, 2), strides=2, padding='same'))
        model.add(Flatten())
        model.add(Dense(number_of_classes, activation='softmax'))
        print('Model: VGG16')
    if cnn == 3:
        baseModel = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=image_size3d)
        if trainable == False:
            for layer in baseModel.layers:
                layer.trainable = False
        headModel = baseModel.output
        headModel = AveragePooling2D(padding='same')(headModel)
        headModel = Flatten()(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.2)(headModel)
        headModel = Dense(number_of_classes, activation='softmax')(headModel)
        model = Model(inputs=baseModel.input, outputs=headModel)
        print('Model: ResNet50')
    if cnn == 4:
        baseModel = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=image_size3d)
        if trainable == False:
            for layer in baseModel.layers:
                layer.trainable = False
        headModel = Flatten()(baseModel.output)
        headModel = Dropout(0.2)(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dense(number_of_classes, activation='softmax')(headModel)
        model = Model(inputs=baseModel.input, outputs=headModel)
        print('Model: Xception')
    if cnn == 5:
        baseModel = tf.keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet',input_shape=image_size3d)
        if trainable == False:
            for layer in baseModel.layers:
                layer.trainable = False
        # model = Sequential()
        # model.add(baseModel)
        # model.add(MaxPool2D((2, 2), strides=2, padding='same'))
        # model.add(Flatten())
        # model.add(Dense(number_of_classes, activation='softmax'))
        #
        headModel = baseModel.output
        headModel = AveragePooling2D(padding='same')(headModel)
        headModel = Flatten()(headModel)
        headModel = Dense(512, activation="relu")(headModel)
        headModel = Dropout(0.4)(headModel)
        headModel = Dense(number_of_classes, activation='softmax')(headModel)
        model = Model(inputs=baseModel.input, outputs=headModel)
        print('Model: DenseNet201')
    if cnn == 6:
        baseModel = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=image_size3d)
        if trainable == False:
            for layer in baseModel.layers:
                layer.trainable = False
        headModel = baseModel.output
        headModel = AveragePooling2D(padding='same')(headModel)
        headModel = Flatten()(headModel)
        headModel = Dense(512, activation="relu")(headModel)
        headModel = Dropout(0.4)(headModel)
        headModel = Dense(number_of_classes, activation='softmax')(headModel)
        model = Model(inputs=baseModel.input, outputs=headModel)
        print('Model: MobileNet')
    # if cnn == 7:
    #     baseModel = tf.keras.applications.EfficientNetB7 (include_top=False, weights='imagenet', input_shape=image_size3d)
    #     if trainable == False:
    #         for layer in baseModel.layers:
    #             layer.trainable = False
    #     headModel = baseModel.output
    #     headModel = AveragePooling2D(padding='same')(headModel)
    #     headModel = Flatten()(headModel)
    #     headModel = Dense(128, activation="relu")(headModel)
    #     headModel = Dropout(0.2)(headModel)
    #     headModel = Dense(number_of_classes, activation='softmax')(headModel)
    #     model = Model(inputs=baseModel.input, outputs=headModel)
    #     print('Model: EfficientNetB2')

    return model

# FOR KERAS TUNER NAS
def build_model(hp):
    # baseModel = tf.keras.applications.InceptionResNetV2(include_top=False, weights='imagenet',
    #                                                     input_shape=image_size3d, classifier_activation='sigmoid')
    headModel = baseModel.output
    if trainable_tune == False:
        for layer in baseModel.layers:
            layer.trainable = False
    print("Trainable tuner = ", trainable_tune)
    headModel = tf.keras.layers.GlobalAveragePooling2D()(headModel)
    headModel = Flatten()(headModel)
    headModel = tf.keras.layers.Dense(hp.Int('units', min_value=32, max_value=512, step=32), activation='relu')(headModel)
    headModel = tf.keras.layers.Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1, default=0.2))(headModel)
    headModel = tf.keras.layers.Dense(number_of_classes, activation='softmax')(headModel)
    model = tf.keras.Model(inputs=baseModel.input, outputs=headModel)
    print(('Tuner'))
    optimizer = hp.Choice('optimizer', values=['adam', 'sgd'], default='adam')
    learning_rate = hp.Choice('learning_rate', values=[0.01, 0.001, 0.005], default=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

    return model

def training(model, cnn, number_of_classes, loss_func, trainGeneratorBuild, valGeneratorBuild):
    callbacks = [
        # tf.keras.callbacks.ModelCheckpoint(
        #     "corn_tranfer_" + "classes_" + str(number_of_classes) + "_Cnn_" + str(cnn) + "_loss_" + str(
        #         loss_func) + ".h5", save_best_only=True, verbose=0),
        tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_loss', verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)]
    print('loss function = categorical_crossentropy')
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall', 'AUC'])

    history = model.fit_generator(trainGeneratorBuild,
                                  validation_data=valGeneratorBuild, epochs=5,
                                  callbacks=[callbacks])
    return history, model


def tuner(input, cnn, data_dir, image_size2d, batch_size):
    print ('Start tuner')
    df = import_dataset(input, data_dir)
    train, test, val, unlabelled, unlabelled_df, train_data = prepare_the_data(number_of_classes=input, df=df, data_dir=data_dir,
                                                   image_size2d=image_size2d, batch_size=batch_size)

    callbacks = [
        # tf.keras.callbacks.ModelCheckpoint(
        #     "corn_tranfer_" + "classes_" + str(number_of_classes) + "_Cnn_" + str(cnn) + "_loss_" + str(
        #         loss_func) + ".h5", save_best_only=True, verbose=0),
        tf.keras.callbacks.EarlyStopping(patience=4, monitor='val_loss', verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)]
    image_size3d = (*image_size2d, 3)
    #model = which_cnn(cnn, number_of_classes=input, image_size3d=image_size3d, trainable=True, tuner=True)

    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=3,
        executions_per_trial=1,
        directory=" ",
        overwrite=True,
        project_name='kerastuner')
    tuner.search_space_summary()

    tuner.search(train, epochs=4, validation_data=val)
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    best_hyperparameters_values = best_hyperparameters.values
    final_model = build_model(best_hps)
    best_model_config = final_model.get_config()
    # Print all the hyperparameters used to create the best model
    for key in best_hyperparameters_values.keys():
        print(f"{key}: {best_hyperparameters_values[key]}")

    history = final_model.fit(train, validation_data=val, epochs=4, callbacks=[callbacks])

    results = final_model.evaluate(test, verbose=1)

    print("test loss", results[0])
    print("test acc:", results[1])
    return history, final_model, results


def overall(input, cnn, loss_function, data_dir, image_size2d, batch_size, trainablee, active_learning):
    df = import_dataset(input, data_dir)
    train, test, val, unlabelled, unlabelled_df, train_data = prepare_the_data(number_of_classes=input, df=df,
                                                                               data_dir=data_dir,
                                                                               image_size2d=image_size2d,
                                                                               batch_size=batch_size)
    image_size3d = (*image_size2d, 3)
    if active_learning == False:
        model = which_cnn(cnn, number_of_classes=input, image_size3d=image_size3d, trainable=trainablee, tuner = False)
        history, model = training(model=model, cnn=cnn, number_of_classes=input, loss_func=loss_function,
                                  trainGeneratorBuild=train, valGeneratorBuild=val)

        results = model.evaluate(test, verbose=1)

        print("test loss", results[0])
        print("test acc:", results[1])
        return history, results, model
    else:
        print(len(train))
        print(len(test))
        print(len(val))
        print(len(unlabelled))
        model = which_cnn(cnn, number_of_classes=input, image_size3d=image_size3d, trainable=trainablee, tuner = False)
        history, model = training(model=model, cnn=cnn, number_of_classes=input, loss_func=loss_function,
                                  trainGeneratorBuild=train, valGeneratorBuild=val)

        results = model.evaluate(test, verbose=1)

        print("test loss", results[0])
        print("test acc:", results[1])

        for i in range(1,3):
            print ("iterration: ", i)
            print(len(unlabelled))

            results = model.evaluate(unlabelled, verbose=1)
            print("accuracy of unlabelled: ", results[1])
            pred_probs = model.predict(unlabelled, verbose=1)
            pred_labels = np.argmax(pred_probs, axis=1)

            unlabelled_predictions = model.predict(unlabelled)

            # calculate the confidence scores for each prediction
            unlabelled_confidence_scores = np.max(unlabelled_predictions, axis=1)

            # Sort the indices of the unlabelled data by confidence score
            sorted_indices = np.argsort(unlabelled_confidence_scores)

            # Select the 100 least confident data points
            least_confident_indices = sorted_indices[:100]
            least_confident_data = unlabelled_df.iloc[least_confident_indices]
            new_unlabelled_df = unlabelled_df.drop(least_confident_data.index)
            print(least_confident_data)
            print(new_unlabelled_df)

            print(least_confident_indices)
            print(least_confident_data)
            # add the 100 least confident data points to the training dataset
            # train_df = pd.read_csv('train_data.csv')
            new_train_df = pd.concat([train_data, least_confident_data], axis=0)
            print(len(new_train_df))

            trainGeneratorBuild = train_data_gen(new_train_df, data_dir,image_size2d,batch_size)

            model = which_cnn(cnn, number_of_classes=number_of_classes, image_size3d=image_size3d, trainable=trainablee, tuner = False)
            history, model = training(model=model, cnn=cnn, number_of_classes=input, loss_func=loss_function,
                                      trainGeneratorBuild=trainGeneratorBuild, valGeneratorBuild=val)

            results = model.evaluate(test, verbose=1)

            print("test loss", results[0])
            print("test acc:", results[1])

            new_unlabelled = validation_gen(new_unlabelled_df,data_dir,image_size2d,batch_size)
            results = model.evaluate(new_unlabelled, verbose=1)
            print("accuracy of unlabelled: ", results[1])
            pred_probs = model.predict(new_unlabelled)
            pred_labels = np.argmax(pred_probs, axis=1)

            unlabelled = new_unlabelled
            unlabelled_df = new_unlabelled_df
            train_data = new_train_df

        return history, results, model


def main():
    global baseModel
    global number_of_classes
    global data_dir
    global trainable_tune
    #number_of_classes = 5
    #data_dir = 'flowers'
    cnn_list = ['VGG16', 'VGG19', 'ResNet50', 'Xception', 'DenseNet121', 'NASNetMobile', 'InceptionV3',
                'EfficientNetB2']
    image_size2d_list = {(224, 224)}
    image_size2d = (224,224)
    image_size3d = (*image_size2d, 3)
    batch_size = 16
    trainable_list = {True}
    active_learning = True

    results_df = pd.DataFrame(columns=['info', 'accuracy_train', 'accuracy_val', 'test accuracy'])
    all_dfs = []
    # image_size2d = (224,224)
    # trainable = True
    acc_train_list = []  # Initialize empty list for acc_train
    acc_no_train_list = []  # Initialize empty list for acc_no_train

    if (active_learning == False):
        min_loss = 1000
        for i in range(1,2):
            for image_size2d in image_size2d_list:
                for trainable in trainable_list:
                    print("image size= ", image_size2d)
                    print("cnn=", i)
                    print("batch size=", batch_size)
                    print("trainable=", trainable)
                    history, results, model = overall(number_of_classes, i, 1, data_dir ,image_size2d, batch_size, trainable_list, active_learning)
                    accuracy_train = []
                    accuracy_val = []
                    accuracy_train.append(history.history['accuracy'])
                    accuracy_val.append(history.history['val_accuracy'])
                    print(accuracy_train)
                    print(accuracy_val)
                    if (results[0] < min_loss):
                        print (results[0], min_loss)
                        min_loss=results[0]
                        best_cnn = i
                    info = f'flowers_transfer_cnn={i}_size={image_size2d}_bs={batch_size}_trainable={trainable}.csv'
                    if len(accuracy_train) != len(accuracy_val):
                        print(
                            f"Error: Length of accuracy_train ({len(accuracy_train)}) does not match length of accuracy_val ({len(accuracy_val)})")
                    else:
                        res = pd.DataFrame({
                            'info': info,
                            'accuracy_train': [item for sublist in accuracy_train for item in sublist],
                            'accuracy_val': [item for sublist in accuracy_val for item in sublist],
                            'test accuracy': results[1],
                            'test loss': results[0]

                        })
                        results_df = results_df.append(res)
                        results_df = results_df.append(pd.Series(), ignore_index=True)
        print ("BEST CNN ", best_cnn)
        print('best loss ', min_loss)
        if (best_cnn==1):
            baseModel = InceptionV3(include_top=False, weights='imagenet', input_shape=image_size3d)
            print("InceptionV3")
        elif (best_cnn==2):
            baseModel = VGG19(include_top=False, weights='imagenet', input_shape=image_size3d)
            print("VGG19")
        elif (best_cnn == 3):
            baseModel = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet', input_shape=image_size3d)
            print("ResNet50V2")
        elif (best_cnn == 4):
            baseModel = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=image_size3d)
            print("Xception")
        elif (best_cnn == 5):
            baseModel = tf.keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet',
                                                                   input_shape=image_size3d)
            print("DenseNet201")
        elif (best_cnn == 6):
            baseModel = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=image_size3d)
            print("MobileNetV2")
        history, final_model, results = tuner(number_of_classes, best_cnn, data_dir, image_size2d, batch_size)
        acc_train = results[1]
        acc_train_list.append(acc_train)


        # print (history.history['accuracy'])
        # print(history.history['val_accuracy'])
        trainable_tune = False
        history1, final_model1, results1 = tuner(number_of_classes, best_cnn, data_dir, image_size2d, batch_size)
        acc_no_train = results1[1]
        acc_no_train_list.append(acc_no_train)

        print("acc_train ", acc_train)
        print("acc_no_train ", acc_no_train)

        if acc_train>acc_no_train:
            best_model = final_model
        else:
            best_model = final_model1

        results_df = results_df.reset_index(drop=True)
        results_df = results_df.append({'acc_train': acc_train, 'acc_no_train': acc_no_train}, ignore_index=True)

        # results_df['acc_train'] = acc_train_list  # Add acc_train as a column to results_df
        # results_df['acc_no_train'] = acc_no_train_list  # Add acc_no_train as a column to results_df

        results_df.to_csv('flowers_tuner.csv', sep=';', index=False)
        results_df.head(20)

    if (active_learning == True):
        print("image size= ", image_size2d)
        print("cnn= 1")
        print("batch size=", batch_size)
        trainable = True
        print("trainable=", trainable)
        cnn = 4
        history, results, model = overall(number_of_classes, cnn , 1, data_dir, image_size2d, batch_size, trainable, active_learning)

if __name__ == "__main__":
    main()
