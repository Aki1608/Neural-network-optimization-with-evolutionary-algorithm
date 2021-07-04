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
from keras.optimizers import adam
import logging


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
    

def feature_map(Hidden_Layers):
    '''
    Gets a list of feature maps according to the number of hidden layer.
    
    Args: 
        int : Number of hidden layer.
    
    Output:
        list: List of feature maps.
    '''
    Feature_Map = {'5':[4,6,10], '7':[4,6,8,12], '9':[4, 6, 8, 10, 14]}
    
    return Feature_Map[f'{Hidden_Layers}']



def Encoder_Decoder_Net(network, generation_no, network_no):
    '''
    Neural network architecture design. 
    
    This function will select the network parameters randomly and create network as a part of 
    evolutionary algorithm. It will then trained and some of the top networks will be selected for
    next generation.
    
    As this is based on U-Net, in the encoder part layers will selected randomly then they will be 
    appended in a list and decoder part will be generated according to the layer list to make sure
    there are no mismatch in encoder decoder part.
    
    skip connection are also included to increase the performance of the network. If it is included
    or not is decided randomly while generating network.
    
    Args:
        network (list): list of parameters that were selected randomly.
        generation_no (int): Number of times to evolve the population.
        network_no (int): in each generation there are fixed number of network which are created.    
    '''
    
    logging.debug('In network function.')
    
    Layers =  (network['Layers'] - ceil(network['Layers'] / 2))
    skip = network['Skip']
    dropout = network['Dropout']
    padding = 'same'
    Feature_Map = feature_map(network['Layers'])
    
    image_height = 144
    image_width = 176
    input_layer = keras.layers.Input((image_height,image_width, 3), name = 'Layer_Input')
    
    Kernel_list = []
    
    strides = 1
    Pooling_layers = 0
    RGB_conv = {}
    RGB_deconv = {}
    skip_connection = {}
    s = 0 #No. of skip connection.
    concatenate = {}
    c = 0 #No. of concatenate layers.
    
    dropout = {}
    d=0 #No. of dropout layer.
    
    Layers = []
    #choice = ['Conv', 'Max_Pool']
    
    ### TO_DO ###
    ## Chnage the choice option selection.
    ### TO_DO ###
    
    choice_1 = ['Conv', 'Max_Pool', 'Max_Pool_4'] 
    '''
    Max_Pool_4: max_pool layer with stride = 4. Only to reduce the size of image by 4 at the start.
                Only use this if there are memory constrains and your image size is too large.
                otherwise go with option "choice".
                
                Will change this later to select this only in the case where this network is used
                for FPGA or microcontroller.
    ''' 
    
    for i in range(network['Layers']-ceil(network['Layers']/2)):
        if i is 0: 
            layer = random.choice(choice_1)
            if layer is 'Conv':
                K_s = random.choice(network['Kernel_Size'])
                Kernel_list.append(K_s)
                RGB_conv[i] = keras.layers.Conv2D(Feature_Map[i], kernel_size=(K_s,K_s), padding=padding, strides=strides, activation="relu", name = f'Layer_RGB_conv_{i}')(input_layer)
                skip_connection[s] = RGB_conv[i]
                s += 1
                Layers.append('Conv')
                
            elif layer is 'Max_Pool':
                RGB_conv[i] = keras.layers.MaxPool2D((2, 2), (2, 2), name = f'Layer_RGB_Pool_{i}')(input_layer)
                Pooling_layers += 1
                Layers.append('Max_Pool')
                
            elif layer is 'Max_Pool_4': #Maxpool layer with stride equals to 4.
                RGB_conv[i] = keras.layers.MaxPool2D((2, 2), (4, 4), name = f'Layer_RGB_Pool4_{i}')(input_layer)
                Pooling_layers += 1
                Layers.append('Max_Pool_4')

            
        elif i >= 1:
            if Pooling_layers < 2 : #Max two pool-layers is allowed.
                layer = random.choice(choice)
                if layer is 'Conv':
                    K_s = random.choice(network['Kernel_Size'])
                    Kernel_list.append(K_s)
                    RGB_conv[i] = keras.layers.Conv2D(Feature_Map[i], kernel_size=(K_s,K_s), padding=padding, strides=strides, activation="relu", name = f'Layer_RGB_conv_{i}')(RGB_conv[i-1])
                    skip_connection[s] = RGB_conv[i]
                    s += 1
                    Layers.append('Conv')
                    RGB_final_layer = RGB_conv[i]

                elif layer is 'Max_Pool':
                    RGB_conv[i] = keras.layers.MaxPool2D((2, 2), (2, 2), name = f'Layer_RGB_Pool_{i}')(RGB_conv[i-1])
                    Pooling_layers += 1
                    Layers.append('Max_Pool')
                    RGB_final_layer = RGB_conv[i]

                else:
                    print('Error: Last layer is not pool or conv.')

            elif Pooling_layers >= 3:
                if dropout == True:
                    K_s = random.choice(network['Kernel_Size'])
                    Kernel_list.append(K_s)
                    dropout[d] = keras.layers.Dropout(0.3)(RGB_conv[i-1])
                    RGB_conv[i] = keras.layers.Conv2D(Feature_Map[i], kernel_size=(K_s,K_s), padding=padding, strides=strides, activation="relu", name = f'Layer_RGB_conv_{i}')(RGB_final_layer)
                    skip_connection[s] = RGB_conv[i]
                    s += 1
                    d += 1
                    Layers.append('Conv')
                    RGB_final_layer = RGB_conv[i]
                    
                else:
                    K_s = random.choice(network['Kernel_Size'])
                Kernel_list.append(K_s)
                RGB_conv[i] = keras.layers.Conv2D(Feature_Map[i], kernel_size=(K_s,K_s), padding=padding, strides=strides, activation="relu", name = f'Layer_RGB_conv_{i}')(RGB_final_layer)
                skip_connection[s] = RGB_conv[i]
                s += 1
                d += 1
                Layers.append('Conv')
                RGB_final_layer = RGB_conv[i]

    #loggin.debug(f'Layers: {Layers}')
    
    #Bottleneck layer(s).
    bottleneck_conv_1 = keras.layers.Conv2D(Feature_Map[-1], kernel_size = (1,1), padding=padding, strides=strides, activation="relu", name = 'Layer_RGB_Bottleneck_0')(RGB_final_layer)

    
    j = 0 #Deconv layer number.
    k = len(Layers)
    if skip == True: #Extra condition to get input from bottleneck layer.
        if Layers[-1] is 'Conv':
            K_s = random.choice(network['Kernel_Size'])
            Kernel_list.append(K_s)
            concatenate[c] = keras.layers.Concatenate(name = f'Layer_RGB_Concatenate_{c}')([skip_connection[s-1], bottleneck_conv_1])
            c += 1
            s -= 1
            RGB_deconv[j] = keras.layers.Conv2D(Feature_Map[-2], kernel_size=(K_s,K_s), padding=padding, strides=strides, activation="relu",name = f'Layer_RGB_Deconv_{j}')(concatenate[c-1])
            j += 1
        elif Layers[-1] is 'Max_Pool':
            RGB_deconv[j] = keras.layers.UpSampling2D((2, 2), name = f'Layer_RGB_Upsampling_{j}')(bottleneck_conv_1)
            j += 1
    else:
        if Layers[-1] is 'Conv':
            K_s = random.choice(network['Kernel_Size'])
            Kernel_list.append(K_s)
            RGB_deconv[j] = keras.layers.Conv2D(Feature_Map[-2], kernel_size=(K_s,K_s), padding=padding, strides=strides, activation="relu",name = f'Layer_RGB_Deconv_{j}')(bottleneck_conv_1)
            j += 1
        elif Layers[-1] is 'Max_Pool':
            RGB_deconv[j] = keras.layers.UpSampling2D((2, 2), name = f'Layer_RGB_Upsampling_{j}')(bottleneck_conv_1)
            j += 1
        
    for i in range(len(Layers)-1):
        if skip == True:
            if Layers[k-2] is 'Conv':
                K_s = random.choice(network['Kernel_Size'])
                Kernel_list.append(K_s)
                concatenate[c] = keras.layers.Concatenate(name = f'Layer_RGB_Concatenate_{c}')([skip_connection[s-1], RGB_deconv[j-1]])
                c += 1
                s -= 1
                RGB_deconv[j] = keras.layers.Conv2D(Feature_Map[k-2], kernel_size=(K_s,K_s), padding=padding, strides=strides, activation="relu", name = f'Layer_RGB_Deconv_{j}')(concatenate[c-1])
                j += 1
                k -= 1
                
            elif Layers[k-2] is 'Max_Pool':
                RGB_deconv[j] = keras.layers.UpSampling2D((2, 2), name = f'Layer_RGB_Upsampling_{j}')(RGB_deconv[j-1])
                j += 1
                k -= 1
                
            elif Layers[k-2] is 'Max_Pool_4':
                RGB_deconv[j] = keras.layers.UpSampling2D((4, 4), name = f'Layer_RGB_Upsampling_{j}')(RGB_deconv[j-1])
                j += 1
                k -= 1
        else:
            if Layers[k-2] is 'Conv':
                K_s = random.choice(network['Kernel_Size'])
                Kernel_list.append(K_s)
                RGB_deconv[j] = keras.layers.Conv2D(Feature_Map[k-2], kernel_size=(K_s,K_s), padding=padding, strides=strides, activation="relu", name = f'Layer_RGB_Deconv_{j}')(RGB_deconv[j-1])
                j += 1
                k -= 1
                
            elif Layers[k-2] is 'Max_Pool':
                RGB_deconv[j] = keras.layers.UpSampling2D((2, 2), name = f'Layer_RGB_Upsampling_{j}')(RGB_deconv[j-1])
                j += 1
                k -= 1
            
            elif Layers[k-2] is 'Max_Pool_4':
                RGB_deconv[j] = keras.layers.UpSampling2D((4, 4), name = f'Layer_RGB_Upsampling_{j}')(RGB_deconv[j-1])
                j += 1
                k -= 1

        
    output = keras.layers.Conv2D(14, (1, 1), padding="same", activation="softmax", name = 'Layer_Output')(RGB_deconv[j-1])
    
    model = keras.models.Model(inputs=input_layer, outputs = output)
    
    return model, Kernel_list
            
#Get number of operations and memory required
def operations_memory (model, network, Kernel_list, generation_no, network_no, time):
    print(f"kernel length: {(Kernel_list)}")
    #From the summery read dimension of all layers and caculate operations and memory usage. 
    with open(f'model_summery/{time}/Model_summery_{generation_no+1}_{network_no+1}.txt','w') as f:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    with open(f'model_summery/{time}/Model_summery_{generation_no+1}_{network_no+1}.txt', 'r') as f:
        lines = f.readlines()

    memory_para_list = []
    MAC_op_para = []
    Layers = []
    Feature_Map = []
    for line in lines[3:-5]:
        if line.startswith('Layer'):
            numbers = re.findall('None, [0-9]+, [0-9]+, [0-9]+', line)
            memory_para_list.append(numbers[0].split(',')[1:])
            
            
        if line.startswith('Layer_RGB_'):    
            Layer = line.split('Layer_RGB_')[1].split('_')[0]
            Layers.append(Layer)
            
            numbers = re.findall('None, [0-9]+, [0-9]+, [0-9]+', line)
            Feature_Map.append(int(numbers[0].split(',')[-1]))

        if line.startswith('Layer_RGB_conv') or line.startswith('Layer_RGB_Deconv') or line.startswith('Layer_RGB_Bottleneck'):
            numbers = re.findall('None, [0-9]+, [0-9]+, [0-9]+', line)
            MAC_op_para.append(numbers[0].split(',')[1:])

            
            
    memory_per_layer = []
    for memory_per_layer_list in memory_para_list:
        memory = 4
        for memory_para in memory_per_layer_list:
            memory *= int(memory_para)
        memory_per_layer.append(memory)
    
    memory_in_use = []
    for i in range(len(memory_per_layer)-1):
        memory_in_use.append(memory_per_layer[i] + memory_per_layer[i+1])
    memory_for_fitness = max(memory_in_use)
    
    mac = 0    
    for j in range(len(Layers)):
        
        ker = 0 
        if Layers[j] == 'conv' or Layers[j] == 'Bottleneck' or Layers[j] == 'Deconv':
            MAC_para = memory_para_list[j+1]
            mac_op = 1
            if Layers[j] == 'conv' or Layers[j] == 'Deconv':
                for i in range(len(MAC_para)):
                    mac_op *= int(MAC_para[i])
                mac_op *= (int(Kernel_list[ker]))**2
                ker += 1
                mac_op *= int(memory_para_list[j][-1])
                mac += mac_op
            
            elif Layers[j] == 'Bottleneck':
                for i in range(len(MAC_para)):
                    mac_op *= int(MAC_para[i])
                mac_op *= int(memory_para_list[j][-1])
                mac += mac_op
    
    
    Mac = (1 + np.exp((mac/10**6) - (3)))**(-1/(4))
    Mem = (1 + np.exp((memory_for_fitness/10**5) - (9)))**(-1/(10))
    
    logging.info('  ')
    logging.info(f'Mac_operatino: {mac/10**6} (max allowable: 2) , Max_memory: {memory_for_fitness/10**5} (max allowable: 8)')
    logging.info(f'Mac: {Mac}, Mem: {Mem}')
    
    fitness_MAC = (1 + np.exp((mac/10**6) - (9)))**(-1/(10)) + (1 + np.exp((memory_for_fitness/10**5) - (9)))**(-1/(10))
    logging.info(f'Fitness part of MAC and Memory: {fitness_MAC}')
    logging.info('  ')
    
    return mac, memory_for_fitness, fitness_MAC, Layers, Feature_Map, memory_per_layer
    

# Generate excel file with architecture of the network.
def architecture_info(network, Layers, Feature_Map, memory_per_layer, mac, generation_no, network_no, time):
    '''
    To generate excel file for each generated network with the info of randomly selected parameters,
    memory usage for each layer and number of operations. 
    
    Args:
        network (list): list of parameters of network.
        Layers (list): list of layers which are selected while creating network.
        Feature_Map (list): list of feature maps for different layers.
        memory_per_layer (int): Memory per layer.
        mac (int): number of operations.
        
        generation_no (int): Number of times to evolve the population
        network_no (int): in each generation there are fixed number of network which are created.    
        time: current time ("current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")") 
    '''
    
    with xlsxwriter.Workbook(f'model_summery/{time}/Network_info_{generation_no+1}_{network_no+1}new.xlsx') as workbook:

        worksheet = workbook.add_worksheet()
        cell_format = workbook.add_format()
        cell_format.set_align('center')
        
        for i in range(len(Layers)*3):
            worksheet.set_row(i, cell_format = cell_format)

        row = 0
        col = 0

        for key, value in network.items():
            worksheet.write(row, col, key)
            col += 1
        
        col = 0
        worksheet.write(row,col+7, 'Memory_usage')
        worksheet.write(len(Layers) + 4, col+2, 'Total_Operations')
        worksheet.write(row+1, col, 'Input')
        
        col = 0
        row = 1
        for i in range(len(Layers)):
            if Layers[i] is 'conv' or 'Deconv'or 'Bottleneck':
                worksheet.write_string(row+1, col, Layers[i])
                worksheet.write_number(row+1,col+1, Feature_Map[i])
                worksheet.write(row+1,col+2, network['Dropout'])
                worksheet.write(row+1,col+3, network['Skip'])
                worksheet.write(row+1,col+4, network['Learning_Rate'])
                worksheet.write(row+1,col+5, network['Batch_Size'])
                row += 1
        
            elif Layers[i] is 'Pool' or 'Upsampling' or 'Concatenate':
                worksheet.write_string(row+1, col, Layers[i])
                worksheet.write_string(row+1,col+1, '-')
                worksheet.write(row+1,col+2, network['Dropout'])
                worksheet.write(row+1,col+3, network['Skip'])
                worksheet.write(row+1,col+4, network['Learning_Rate'])
                worksheet.write(row+1,col+5, network['Batch_Size'])
                row += 1      
                
        col = 0
        worksheet.write(len(Layers)+2, col, 'Output')
        
        row = 0
        col = 0
        for i in range(len(memory_per_layer)):
            worksheet.write(row+1,col+7, memory_per_layer[i])
            row += 1
        
        worksheet.write(len(Layers) + 4, col+4, mac)
        
    return
    
def compile_model(network, generation_no, network_no, time):
    
    '''
    Compile the model.
    
    Args:
        network (list): list of parameters of network.
        generation_no (int): Number of times to evolve the population
        network_no (int): in each generation there are fixed number of network which are created.    
        time: current time ("current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")")
    
    '''
    
    logging.debug('In compile network function.')
    
    tf.keras.optimizers.Adam( 
        learning_rate=network['Learning_Rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='adam')

    model, Kernel_list = Encoder_Decoder_Net(network, generation_no, network_no)
    model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["acc"])
    model.summary()
    
    mac, memory_for_fitness, fitness_MAC, Layers, Feature_Map, memory_per_layer = operations_memory(model, network, Kernel_list, generation_no, network_no, time)
    architecture_info(network, Layers, Feature_Map, memory_per_layer, mac, generation_no, network_no, time)
    
    return model, fitness_MAC, Kernel_list, Layers, (mac/10**6), (memory_for_fitness/10**5)


def train_and_accuracy(network, generation_no, network_no, time):   
    
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

    model, MAC_fitness, Kernel_list, Layers, Mac, Mem = compile_model(network, generation_no, network_no, time)

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
    with open(f'model_summery/{time}/model_{generation_no+1}_{network_no+1}.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(f'model_summery/{time}/model_{generation_no+1}_{network_no+1}.h5')
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