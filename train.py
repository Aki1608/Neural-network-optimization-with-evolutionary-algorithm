#!Python3

##Imports
import os
import re
import sys
import random
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import xlsxwriter
import csv
import datetime

import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import keras
from keras.optimizers import Adam
import logging

import utils

logging.debug('In train file.')

#Generation of batches of images and masks
class DataGen(keras.utils.Sequence):
    def __init__(self, image_names, path, batch_size, image_width = 176, image_height = 144):
        self.image_names = image_names
        self.path = path
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        
    def __load__(self, image_name):
        # Path
        image_RGB_path = os.path.join("../../DataRecording/New_Recording/image_processing/All_images", self.path, image_name) + ".png"
        mask_path = os.path.join("../../DataRecording/New_Recording/image_processing/All_masks_txt", self.path, image_name) +  ".txt"
        
        # Reading Image
        image_RGB = cv2.imread(image_RGB_path, 1)
        image_RGB = cv2.resize(image_RGB, (self.image_width,self.image_height))
        image_array = np.array(image_RGB)

        
        # Reading Mask
        with open(mask_path) as f:
            txt = f.read()
            mask = txt.split('\n')
            mask_final = []
            for i in range(len(mask)):
                mask_2 = (mask[i].split(','))
                mask_final.extend(mask_2)
            mask_final.remove('')
            mask_array = np.array(mask_final)
            mask = np.reshape(mask_array, (self.image_height,self.image_width))
            mask = tf.keras.utils.to_categorical(mask, 14)


        
        ## Normalizaing 
        image_RGB = image_RGB/255.0

        
        return image_RGB,mask
    
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.image_names):
            self.batch_size = len(self.image_names) - index*self.batch_size
        
        image_batch = self.image_names[index*self.batch_size : (index+1)*self.batch_size]
        
        image_RGB = []
        mask  = []
        
        for image_name in image_batch:
            _img_RGB, _mask = self.__load__(image_name) #_img_Depth, _mask = self.__load__(image_name)
            
            image_RGB.append(_img_RGB)
            mask.append(_mask)
            
        image_RGB = np.array(image_RGB)
        mask  = np.array(mask)
        
        return image_RGB, mask
    
    def __len__(self):
        return int(np.ceil(len(self.image_names)/float(self.batch_size)))
    
    
    
def train_and_accuracy(network: list, generation_no: int, network_no: int, time: str):   
    
    '''
    To train the randomly created network and find get accuracy values.
    
    Args:
        network (list): list of parameters of network.
        generation_no (int): Number of times to evolve the population.
        network_no (int): in each generation there are fixed number of network which are created.    
        time: current time ("current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")")
    '''
    
    logging.debug('In train and accuracy function.')
    
    train_path = "Train"
    test_path = "Test"

    epochs = 30
    batch_size = network['Batch_Size']

    image_height = 144
    image_width = 176
   
    # Training and test Images
    train_images = os.listdir('../../DataRecording/New_Recording/image_processing/All_images/Train')
    test_images = os.listdir('../../DataRecording/New_Recording/image_processing/All_images/Test')

    train_img = []
    for image in train_images:
        image1 = image.split('.')
        train_img.append(image1[0])
    #print(f'Length of training set: {len(train_img)}')

    test_img = []
    for image in test_images:
        image1 = image.split('.')
        test_img.append(image1[0])
    #print(f'Length of test set: {len(test_img)}')
    
    ## Validation Data Size
    val_data_size = 600

    valid_img = train_img[:val_data_size]
    train_img = train_img[val_data_size:]
        
    train_gen = DataGen(train_img, train_path, image_height=image_height, image_width=image_width, batch_size=batch_size)
    valid_gen = DataGen(valid_img, train_path, image_height=image_height, image_width=image_width, batch_size=batch_size)

    train_steps = len(train_img)//batch_size
    valid_steps = len(valid_img)//batch_size

    model, MAC_fitness, Kernel_list, Layers, Mac, Mem = utils.compile_model(network, generation_no, network_no, time)

    #Tensorboard
    '''tb_callback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    
    history = model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=epochs, callbacks = [tb_callback])'''
    
    history = model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=epochs)
    
    accuracy = history.history['acc']
    loss = history.history['loss']
    
    acc,los = [], []
    for acc_value in accuracy:
        acc.append(f'{acc_value: 0.4f}')
    for los_value in loss:
        los.append(f'{los_value: 0.4f}')
        
    logging.info(f'Accuracy and loss at each epoch: {acc}, {los}')
    
    # Save the model.
    model_json = model.to_json()
    with open(f'model_summary/{time}/model_{generation_no+1}_{network_no+1}.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f'model_summary/{time}/model_{generation_no+1}_{network_no+1}.h5')
    logging.info("Saved model to disk")
    
    # Generate test dataset.
    test_gen = DataGen(test_img, test_path, image_width=176, image_height=144, batch_size=batch_size)
    
    # Accuracy for test dataset.
    eval_accuracy = 0
    for i in range(len(test_img)//batch_size):
        x,y = test_gen.__getitem__(i) 
        score = model.evaluate(x,y)
        eval_accuracy += score[1]
    eval_accuracy /= (len(test_img)/batch_size)    
    
    
    logging.info(f'Evaluation accuracy: {eval_accuracy: 0.4f}, MAC and Memory: {MAC_fitness}')
    
    return total_score, 1*(MAC_fitness), Kernel_list, Layers, Mac, Mem