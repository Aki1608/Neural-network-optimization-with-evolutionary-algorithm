#!/usr/bin/env python
# coding: utf-8

import xlsxwriter

import random
import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

import os
if not os.path.exists(f"./model_summary/{current_time}"):
    os.makedirs(f"model_summary/{current_time}")


import logging
logging.basicConfig(filename=f'model_summary/{current_time}/example.log', format='%(asctime)s %(levelname)-8s %(message)s', 
                    level = logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from EA_Optimizer import Optimizer
from tqdm import tqdm
from network import Network

logging.debug('In main file.')

def train_networks(networks: list,generation_no: int, time: str) -> None:
    '''
    Train each network.
    
    Args:
        networks (list): list of parameters of network
        generation_no (int): generation number
        time (str): time in "%Y%m%d-%H%M%S" format
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


def get_average_accuracy(networks: list) -> float:
    
    '''
    Get the average segmentation accuracy of neural networks for each generation.
    
    Args:
        networks (list): list of network parametrs.
        
    '''
    
    logging.debug('In average accuracy in main. calculating.')
    
    total_accuracy = 0
    for network in networks:
        total_accuracy += network.accuracy
    
    logging.debug('Accuracy calculated.')
    
    return total_accuracy / len(networks)


def generate(generations: int, population: int, para_choice: list, time: str) -> None:
    
    logging.debug('In generate function. In main file.')
    '''
    Generate a network with the genetic algorithhm.
        
    Args:
        generations (int): Number of times to evolve the population
        population (int): Number of networks in each generaion
        para_choice (dit): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating
    '''
    
    optimizer = Optimizer(para_choice)

    networks = optimizer.create_population(population)
    
    for generation_no in range(generations):
        logging.info(f'***Training generation {generation_no+1} of {generations}')
        
        logging.info(f'Length of population before generation: {generation_no+1} is {len(networks)}')
        
        train_networks(networks,generation_no, time)
        
        average_accuracy = get_average_accuracy(networks)
                
        logging.info(f'Generation average: {average_accuracy:.2f}')
        logging.info('-'*80)
        
        if generation_no != generations - 1:
            #Do the evolution.
            logging.info('Evolving.')
            networks = optimizer.evolve(networks)
            
            
    #Sort our final population.
    networks = sorted(networks, key=lambda x:x.accuracy, reverse=True)
    
    #Print out the all networks.
    print_networks(networks[:])

    
def print_networks(networks: list) -> None:
    '''
    Print a list of networks.
    
    Args:
        networks (list): The population of networks
    '''
    logging.info('-'*80)
    for network in networks:
        network.print_network()
        

def main() -> None:    
    generations = 10
    population = 12
    
    para_choice = {'Layers': [5,7,9],
             'Kernel_Size': [[1,3,5,7], [1,3], [1,5], [1,7], [1,3,7], [3,5,7], [5,1,3]],
             'Dropout': [True, False],
             'Skip' : [True, False],
             'Learning_Rate': [0.01, 0.001, 0.02, 0.0005, 0.005],
             'Batch_Size': [4,5,6,7,8]}
                   
    logging.info(f'***Evolving {generations} generations with population {population}')

    generate(generations, population, para_choice, current_time)
    
    
if __name__ == '__main__':
    main()

