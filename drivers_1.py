import os
import numpy as np
import pandas as pd
import time
import cv2
import random
import pickle
import matplotlib.pyplot as plt
import h5py
os.chdir(r'D:/drivers/')

from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers.core import Dense, Dropout, Flatten
from tensorflow.contrib.keras.python.keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.contrib.keras.python.keras.layers.pooling import AveragePooling2D
from tensorflow.contrib.keras.python.keras.layers import InputLayer
from tensorflow.contrib.keras.python.keras.models import load_model
from tensorflow.contrib.keras.python.keras import optimizers
from tensorflow.contrib.keras.python.keras.callbacks import EarlyStopping
from tensorflow.contrib.keras.python.keras.callbacks import ModelCheckpoint
from tensorflow.contrib.keras.python.keras.callbacks import History
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras.layers.core import Lambda
import tensorflow as tf
K.set_learning_phase(0)


def load_image(filepath, size):
    """"args: filepath, image size
    
    Uses OpenCV package to transform image into numerical format.
    
    Returns: list of shape size*size*3
    """
    image=cv2.imread(filepath)
    res=cv2.resize(image,(size,size),cv2.INTER_LINEAR)
    return res

def load_train_data(size, fp = 'D:\\drivers\\train'):
    """args: image size, filepath of training data location
    
    Loads the training data from the filepath and converts it to numerical format.
    
    Returns 3 lists: 
        1. training features, shape = images*size*size*3
        2. training labels, shape = images*1
        3. pciture ids< shape  = images * 1
    """
    os.chdir(fp)
    cwd=os.getcwd()
    folders=[name for name in os.listdir(cwd) if os.path.isdir(cwd+'\\'+name)]
    
    x_train=[]
    y_train=[]
    picture_ids=[]
    start=time.time()
    for folder in folders:
        fp=cwd+'\\'+folder
        for item in os.listdir(fp):
            img_fp=fp + '\\' + item
            img=load_image(img_fp, size)
            x_train.append(img)
            y_train.append(folder)
            picture_ids.append(item)
    
    end=time.time()
    print('Train data load time is {}'.format(end-start))
    print('Number of training images is {}'.format(len(picture_ids)))
    return x_train,y_train,picture_ids

def load_test_data(size, fp = 'D:\\drivers\\test'):
    """args: image size, filepath of testing data location
    
    Loads the testing data from the filepath and converts it to numerical format.
    
    Returns 2 lists: 
        1. testing features, shape = images*size*size*3
        2. pciture ids< shape  = images * 1
    """
    x_test=[]
    picture_ids=[]
    start=time.time()
    for item in os.listdir(fp):
        img_fp=fp + '\\' + item
        img=load_image(img_fp, size)
        x_test.append(img)
        picture_ids.append(item)
    end=time.time()
    print('Test data load time is {}'.format(end-start))
    print('Number of test images is {}'.format(len(picture_ids)))
    return x_test,picture_ids

def load_driver_data(fp = 'D://drivers//' ):
    """
    Args: filepath of driver data.
    
    Loads the data mapping drivers to the list of their images.
    
    Returns: a dictionnary with a key for each driver and a value with
    the list of their images.
    """
    os.chdir(fp)
    data = pd.read_csv("driver_imgs_list.csv")
    drivers = {}
    for driver in data['subject'].unique():
        if driver not in drivers:
            temp = data[data['subject'] == driver]
            drivers[driver] = list(temp['img'])
    return drivers

def process_labels(labels):
    """
    Args: list of image labels.
        
    The labels come as categories c0 - c9. This function converts categorical class 
    labels into integer values 0-9.
    
    Returns: array of labels with integer values. Shape = images * 1.
    """
    b,c=np.unique(labels,return_inverse=True)
    return c

def process_features(features):
    """
    Args: list of images features.
    
    Converts the list into a numpy array and divides all featurs by 255, so that
    pixel values range from 0 to 1 instead of 0 to 255.
    
    Returns: array of scaled features. Shape = images * size * size * 3.
    """
    X = np.array(features)
    try:
        X = X.reshape(X.shape[0],X[0].shape[0], X[0].shape[1], 3)
    except:
        X = X.reshape(1,X.shape[0], X.shape[1], 3)
    X = X/255
    return X

def summary_stats(labels):
    """
    Args: training labels.
    
    Prints 2 summary statistics for the training data: the most common image and
    the frequency with which it occurs.
    """
    from collections import Counter
    cnt=Counter()
    for label in labels:
        cnt[label]+=1
    print('Most common image is {}'.format(cnt.most_common(n=1))),
    print(' with {}% of the total'.format(100*max(cnt.values())/len(labels)))

def split_by_driver(X,y, id_train,drivers_dict):
    """
    Args: list of features, list of labels, list of image ids and dictionnary mapping
    drivers to list of their images.
    
    Saves to memory an arrays of images for each driver that contains the images that 
    driver is in. The arrays are named after the driver id.
    """
    
    os.chdir(r'D:\drivers\imgs_224')
    start=time.time()
    for driver in list(drivers_dict):
        print(driver)
        temp=drivers_dict[driver]
        X_temp = np.array([X[i] for i in range(len(id_train)) if id_train[i] in temp])
        y_temp = np.array([y[i] for i in range(len(id_train)) if id_train[i] in temp])
        X_mem = np.memmap('X_{}'.format(driver), dtype='float32', mode='w+', shape=(X_temp.shape))
        y_mem = np.memmap('y_{}'.format(driver), dtype='float32', mode='w+', shape=(y_temp.shape))
        X_mem[:] = X_temp[:]
        y_mem[:] = y_temp[:]
        X_mem.flush()
        y_mem.flush()
    end=time.time()
    print('Time to split is {}'.format(end-start))
    
def train_valid_split_by_driver(drivers_dict, id_train, num_valid=4):
    """Args: dictionnary mapping drivers to the list of their images, list of
    image ids and the number of drivers to include in the validation set.
    
    Saves to memory training data and validation data. The split between training set
    and the validation set is not random as they are determined by which driver is in the
    photo. The final num_valid drivers are used for the validation set, the rest of the
    drivers are used for the training set.
    """
    os.chdir(r'D:/drivers/imgs_224') 
    drivers = list(drivers_dict)
    valid = drivers[-num_valid:]
    train = drivers[:- num_valid]
    
    num_valid = 0
    for driver in drivers:
        if driver in valid:
            num_valid += len(drivers_dict[driver])#
    num_train = 22424 - num_valid
    
    X_train = np.memmap("X_train", dtype='float32', mode='w+', shape=(num_train,224,224,3))
    X_valid = np.memmap("X_valid", dtype='float32', mode='w+', shape=(num_valid,224,224,3))
    y_train = np.memmap("y_train", dtype='float32', mode='w+', shape=(num_train))
    y_valid = np.memmap("y_valid", dtype='float32', mode='w+', shape=(num_valid))

    
    train_start=0
    valid_start=0
    for driver in drivers:
        X_temp = np.memmap("X_{}".format(driver), dtype='float32',mode='r',shape=(len(drivers_dict[driver]),224,224,3))
        y_temp = np.memmap("y_{}".format(driver), dtype='float32',mode='r',shape=(len(drivers_dict[driver])))
        if driver in train:
            X_train[train_start:train_start+len(y_temp)] = X_temp
            y_train[train_start:train_start+len(y_temp)] = y_temp
            train_start+=len(y_temp)
        else:
            X_valid[valid_start:valid_start+len(y_temp)] = X_temp
            y_valid[valid_start:valid_start+len(y_temp)] = y_temp
            valid_start+=len(y_temp)
        X_temp.flush()
        y_temp.flush()
    X_train.flush()
    X_valid.flush()
    y_train.flush()
    y_valid.flush()

def make_vgg16(top = True, weights='imagenet', weight_path=''):
    """
    Args: top determines whether to include dense layers at the top. Weights determines
    whether to use imagenet weights or pre-trained weights, in which case the filepath
    must be specified via weight_path.
    
    Creates a convolutional neural network following the VGG16 structure. There are two options:
    the original structure with fully connected layers at the end, or a slimmed down model where the 
    FC layers are replaced by a global average pooling layer. The latter has far fewer weights.
    
    Returns the model.
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224,224,3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    if weights=='imagenet':
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='softmax'))
        model.load_weights(r'D:/drivers/vgg16_weights.h5')
        if top==False:
            for i in range(7):
                model.layers.pop()
            model.outputs = [model.layers[-1].output]
            model.layers[-1].outbound_nodes = []
            model.add(Convolution2D(1024, (3, 3), activation='relu'))
            model.add(AveragePooling2D((14,14),padding='same'))
            model.add(Flatten())
            model.add(Dense(10, activation='softmax'))
        else:
            model.layers.pop()
            model.outputs = [model.layers[-1].output]
            model.layers[-1].outbound_nodes = []
            model.add(Dense(10, activation='softmax'))
    elif weights=='trained':
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        if top==False:
            for i in range(7):
                model.layers.pop()
            model.outputs = [model.layers[-1].output]
            model.layers[-1].outbound_nodes = []
            model.add(Convolution2D(1024, (3, 3), activation='relu'))
            model.add(AveragePooling2D((14,14),padding='same'))
            model.add(Flatten())
            model.add(Dense(10, activation='softmax'))
        model.load_weights(weight_path)
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
    return model
    
def submit(model, X_upload, id_upload, fp):
    """
    Args: model to classify images, array of test features, array of test ids,
    filepath to save predictions to in a csv.
    
    Classifies a list of test images then saves the predictions in the correct format
    for Kaggle upload in a csv file.
    
    Returns: DataFrame of predictions and creates csv file.
    """
    os.chdir('D:\\drivers\\')
    #X_upload = process_features(X_upload)
    print(len(X_upload), len(id_upload))
    start = time.time()
    label_upload = model.predict(X_upload)
    end = time.time()
    print('Time to predict {} images is {}s'.format(len(label_upload), end-start))
    
    classes = ['c' + str(i) for i in range(10)]
    submission = pd.DataFrame(label_upload, columns = classes)
    submission.loc[:, 'img'] = pd.Series(id_upload, index = submission.index)
    submission.to_csv(fp, index=False)
    return submission


def show_image(filepath):
    """Args: filepath of an image.
    
    Uses OpenCV to show the image in a separate window. The image will disappear once
    0 is pressed.
    """
    img=cv2.imread(filepath)
    cv2.imshow(filepath,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_image_pred(filepath, model, size=224, show=True):
    """
    Args: Filepath of image, model to predict with and size of the image that the model requires.
    
    Shows an image and prints the model's classification of the image.
    """
    CLASS_INDEX = {0 : 'safe', 1: 'text right hand', 2: 'phone right hand', 3: 'text left hand', 4:'phone left hand', 5: 'radio',
               6: 'food/drink', 7: 'reaching behind', 8:'hair/makeup', 9: 'talking to passenger'}
    X = load_image(filepath, size)
    X = process_features(X)
    logits = model.predict(X)
    if show:
        print(CLASS_INDEX[np.argmax(logits)] + '_with probability' + str(np.round(np.argmax(logits)),2))
        show_image(filepath)

def predict_test_sample(model, N, show=True, size=224):
    """
    Args: model to make predictions, sample size, whether to show the image,
    size of images to predict (height & width)
    
    Makes predictions using specified model for a sample of N images. 
    Will show the images if show=True. To move to next image, press 0.
    """
    images = os.listdir('D://drivers//test')
    start = time.time()
    for i in range(N):
        show_image_pred('D://drivers//test//'+random.choice(images), model, size, show)
    end = time.time()
    if show == False:
        print(np.round((end-start)/float(N),4))

def next_batch(start,train,labels,batch_size=2803):
    """
    Args: batch start id, train data, image labels and batch size.
    
    Generates a list of ids for a given batch of the train data. Used for pre-processing
    images where the data is too large to fit in RAM, so process by batch instead.
    
    Returns: train data for the batch, ids for the batch and the start position
    for the next batch.
    """
    train_ids = [i for i in range(train.shape[0])]
    newstart = start+batch_size
    if newstart > train.shape[0]:
        newstart = 0
    ids = train_ids[start:start+batch_size]
    return train[ids,:], labels[ids,:], newstart

def generator(features,labels,batch_size):
    """
    Args: features of data, labels of data and the batch size for training.
    
    Used in Keras' fit_generator function for training a network, where the training data
    would be too large for the RAM. Instead feeds only the training data for a given 
    batch to the algorithm, which is stored only temporarily.
    
    Returns: featurs of the batch, labels of the batch.
    """
    batch_features = np.zeros((batch_size,224,224,3))
    batch_labels = np.zeros((batch_size,1))
    while True:
        for i in range(batch_size):
            index = random.choice(range(len(features)))
            batch_features[i]=features[index]
            batch_labels[i] = labels[index]
        yield (batch_features,batch_labels)

def process_by_batch(features, filepath, batch_size=2803):
    """
    Args: features of data, filepath to save to, size of batch for processing.
    
    Used when dataset is too large to fit in RAM, therefore cannot all be processed
    simultatenously. Instead uses Numpy's memap class to process the data in smaller batches.
    Memap allows us to temporarily store only the batch that is being processed,
    rather than the entire dataset.
    
    Saves processed data to the filepath.
    """
    X_mem = np.memmap(filepath, dtype='float32', mode='w+', shape=(79726,224,224,3))
    s = time.time()
    start = 0
    while True:
        if start>=len(features):
            break
        end=start+min(batch_size, len(features)-start)
        print(start,end)
        X = features[start:end]/255.0
        X_mem[start:end] = X[:]
        start += batch_size
    e = time.time()
    X_mem.flush()
    print('time taken to process is {}'.format(e-s))


def target_category_loss(x, category_index, nb_classes):
    """
    Used in heatmap function
    """
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def normalize(x):
    """
    Used in heatmap function - normalises a tensor by its L2 norm.
    """
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image_cam(fp):
    """
    Args: filepath for image.
    
    Used in heatmap function. Loads an image then returns its features after 
    having been normalised to a [0,1] range.
    
    Returns: numpy array of normalised image features.
    """
    x = cv2.imread(fp).astype('float32')
    x = cv2.resize(x,(224,224),cv2.INTER_LINEAR)
    x[:, :, 0] /= 255.0#103.939
    x[:, :, 1] /= 255.0#116.779
    x[:, :, 2] /= 255.0#123.68
    x=np.expand_dims(x,axis=0)
    return x

def decode_predictions(preds, top=1):
    """
    Args: array of predictions, number of predictions to print.
    
    Takes an array of predictions and returns the most likely predicted classes and
    their respective probabilities.
    
    Returns: List of 3 items: the class number, the name of the class and its 
    predicted probability.
    """
    
    CLASS_INDEX = {0 : 'safe', 1: 'text right hand', 2: 'phone right hand', 3: 'text left hand', 4:'phone left hand', 5: 'radio',
               6: 'food/drink', 7: 'reaching behind', 8:'hair/makeup', 9: 'talking to passenger'}
    top_indices = preds.argsort()[0][-top:][::-1][0]
    return [[CLASS_INDEX[top_indices], str(top_indices), preds[0][top_indices]]]


def grad_cam(input_model, image, category_index):
    """
    Args: model to make predictions, image to predict, index of categories and
    their predicted probabilities.
    
    Constructs a colour map showing where the classifier puts the highest weight
    for a given image in making its prediction.
    
    Returns: numpy array of same dimension as image but instead displaying colours
    according to where the classifier puts the most weight.
    """
    model = Sequential()
    model.add(input_model)
    nb_classes = 10
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    model.add(Lambda(target_layer))
    loss = K.sum(model.layers[-1].output)
    conv_output =  model.layers[0].layers[29].output #this needs changed depending on NN structure
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)
    
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255.0*cam / np.max(cam)
    return np.uint8(cam)

def show_heatmap(img_fp, text):
    """
    Args: filepath of image, text to display on image.
    
    Similar to the show image function but specifically for heatmaps. Adds a box
    to the image showing the predicted class and probability.
    
    Only displays the image in the interpreter, does not save or return.
    """
    features = cv2.imread(img_fp)
    b,g,r = cv2.split(features)       # get b,g,r
    features = cv2.merge([r,g,b])     # switch it to rgb
    fig = plt.figure()
    ax = fig.add_subplot(111)
    bbox_props = dict(boxstyle='Round', fc="red", ec="b", lw=2)
    ax.text(10, 10, text,
            size=10,
            bbox=bbox_props)
    plt.axis('off')
    plt.imshow(features)
    plt.show()

def get_heatmap(out_fp = r'D:/drivers/cam_test.jpg',img_fp ='', model = ''):
    """Args: filepath to save heatmap, filepath of image to generate heatmap from,
    model to make the prediction. If img_fp is blank then selects a random image
    from the unlabelled testing images.
    
    Combines many heatmap functions to predict the class of an image, then creates a heatmap
    showing where the classifier puts the most weight.
    
    Saves the heatmap to out_fp, then displays the heatmap annotated with a box
    displaying predicted class and probability.
    """
    if img_fp == '':
        img_fp = r'D:/drivers/test/' + random.choice(os.listdir(r'D:/drivers/test/'))
    if model=='':
        model = make_vgg16(weights='trained',weight_path=r'D:/drivers/imgs_224/vgg16_valid_1.h5')
    print('fp is {}'.format(img_fp))
    preprocessed_input = load_image_cam(img_fp)
    predictions = model.predict(preprocessed_input)
    top_1 = decode_predictions(predictions)[0]
    print('Predicted class:')
    print('%s (%s) with probability %.2f' % (top_1[1], top_1[0], top_1[2]))
    
    predicted_class = np.argmax(predictions)
    cam = grad_cam(model, preprocessed_input, predicted_class)
    img_orig = cv2.imread(img_fp)
    cam_out = cv2.resize(cam,(img_orig.shape[1], img_orig.shape[0]))
    img = 0.5*cam_out + img_orig
    cv2.imwrite(out_fp, img)
    show_heatmap(out_fp, text = '%s, p=%.2f' % (top_1[0],top_1[2]))

def compare_heat(model1, model2,img_fp=''):
    """
    Args: 2 models to compare heatmaps. Filepath of image to generate heatmap. If
    img_fp is blank then selects a random image from the unlabelled testing data.
    
    Does not save or return, only displays two heatmaps annotated with the
    predicted class and probability from each model.
    """
    if img_fp == '':
        img_fp = r'D:/drivers/test/' + random.choice(os.listdir(r'D:/drivers/test/'))
    get_heatmap(out_fp=r'D:/drivers/cam_{}.jpg'.format(model1.name), img_fp=img_fp,model=model1)
    get_heatmap(out_fp=r'D:/drivers/cam_{}.jpg'.format(model2.name), img_fp=img_fp,model=model2)


if __name__=='__main__':
    os.chdir('D:\\drivers\imgs_224')  #CHANGE TO RELEVANT DIRECTORIES THAT YOU WANT TO SAVE FILES IN
    drivers_dict=load_driver_data()
    X_all, y_all, id_all = load_train_data(224,fp='D:\\drivers\\train') #loads the training data
    X_upload, id_upload = load_test_data(size = 224, fp = 'D:\\drivers\\test')
    
    
    process_by_batch(X_all, 'X_all') #process the training features and saves to the given filepath
    process_by_batch(X_upload,'X_upload_processed') #process the testing features
    y_all = process_labels(y_all) #process the training labels
    X_all = np.memmap("X_all", dtype='float32', mode='r', shape=(22424,224,224,3)) #loads processed data
    split_by_driver(X_all,y_all,id_all) #saves files of features and labels for each driver. names of files given by driver id.
    
    #splits the training data into training set and validation set. Sizes are given by number of images for last 4 drivers, which is the validation set.
    train_valid_split_by_driver(drivers_dict,id_all,num_valid=4) 
    X_train = np.memmap("X_train", dtype='float32', mode='r', shape=(18654,224,224,3)) 
    X_valid = np.memmap("X_valid", dtype='float32', mode='r', shape=(22424-18654,224,224,3))
    y_train = np.memmap("y_train", dtype='float32', mode='r', shape=(18654))
    y_valid = np.memmap("y_valid", dtype='float32', mode='r', shape=(22424-18654))
    
    #load the test data into numpy memap
    X_upload = np.memmap("X_upload_processed", dtype='float32', mode='r', shape=(79726,224,224,3))
    
    #build the models
    vgg = make_vgg16(weights ='imagenet', top=True)
    no_FC = make_vgg16(weights = 'imagenet', top = False)
    vgg.name = 'vgg'
    no_FC.name = 'no_FC'
    
    #train the models, then upload submission to kaggle
    early_stopping = EarlyStopping(monitor='val_loss', patience=0)
    for model in [vgg, no_FC]:
        model.fit_generator(generator(X_train,y_train,32), steps_per_epoch=int(len(X_train)/32),
        validation_data=(X_valid,y_valid),epochs=5,callbacks = [early_stopping, History()])
        model.save(r'D:/drivers/imgs_224/{}'.format(model.name))
        submit(model,X_upload, 'submission_{}'.format(model.name))
        get_heatmap(model = model) #generate a heatmap from each model
    
    #generate some heatmaps comparing the models
    for i in range(3):
        compare_heat(vgg,no_FC)
    
    
    