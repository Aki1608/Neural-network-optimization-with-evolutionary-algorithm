#!/usr/bin/env python
# coding: utf-8

import xlsxwriter

import random
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

import os
os.mkdir(f'model_summery/{current_time}')

import logging
logging.basicConfig(filename=f'model_summery/{current_time}/example.log', format='%(asctime)s %(levelname)-8s %(message)s', 
                    level = logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from EA_Optimizer_GPU import Optimizer
from tqdm import tqdm
from network_GPU import Network

logging.debug('In main file.')

def train_networks(networks,generation_no, time):
    '''
    Train each network.
    
    Args:
        network (list): Current population of networks
        dataset (str): Dataset to use for training/evaluating
    '''
    for i in range(len(networks)):
        Network.print_network(networks[i])
    
    pbar = tqdm(total=len(networks))
    network_no = 0
    for network in networks:
        logging.info('  ')
        logging.info('  ')
        logging.info(f'Generation: {generation_no+1}, population:{network_no+1}')
        Network.print_network(networks[network_no])
        network.train(generation_no, network_no, time)
        pbar.update(1)
        network_no += 1
    pbar.close()


def get_average_accuracy(networks):
    
    logging.debug('In average accuracy in main. calculating.')
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy
    
    logging.debug('Accuracy calculated.')
    
    return total_accuracy / len(networks)


def generate(generations, population, para_choice, time):
    
    logging.debug('In generate function. In main file.')
    '''
    Generate a network with the genetic algorithhm.
        
    Args:
        generations (int): Number of times to evolve the population
        population (int): Number of networks in each generaion
        para_choice (dit): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating
    '''
    
    logging.debug('Before optimizer. IN main.')
    optimizer = Optimizer(para_choice)
    logging.debug('After Optimizer.')
    
    
    logging.debug('Before network. in Main.')
    networks = optimizer.create_population(population)
    logging.debug('After network. in main.')
    
    for generation_no in range(generations):
        logging.info(f'***Training generation {generation_no+1} of {generations}')
        logging.info(f'Length of population before generation: {generation_no+1} is {len(networks)}')
        
        train_networks(networks,generation_no, time)
        
        logging.debug('After train network.')
        
        average_accuracy = get_average_accuracy(networks)
        
        logging.debug('Back after accuracy in main.')
        logging.info(f'Generation average: {average_accuracy:.2f}')
        logging.info('-'*80)
        
        if generation_no != generations - 1:
            #Do the evolution.
            logging.info('Evolving.')
            networks = optimizer.evolve(networks)
            
            
    #Sort our final population.
    networks = sorted(networks, key=lambda x:(x.accuracy + x.mac_op) , reverse=True)
    
    #Print out the all networks.
    print_networks(networks[:])

    
def print_networks(networks):
    '''
    Print a list of networks.
    
    Args:
        networks (list): The population of networks
    '''
    logging.info('-'*80)
    for network in networks:
        network.print_network()
        

def main():    
    generations = 10
    population = 15
    
    para_choice = {'Layers': [9,11,13],
             'Kernel_Size': [[1,3,5,7], [1,3], [1,5], [1,7], [1,3,7], [3,5,7], [5,1,3], [5,7,1], [3,5], [3,7], [5,7]],
             'Dropout': [True, False],
             'Skip' : [True, False],
             'Learning_Rate': [0.01, 0.001, 0.02, 0.0005, 0.005],
             'Batch_size': [4,5,6,7,8]}
                   
    logging.info(f'***Evolving {generations} generations with population {population}')

    generate(generations, population, para_choice, current_time)
    
    
if __name__ == '__main__':
    main()

