#! Python3

#import Main
import random
import logging
from train_final_GPU import train_and_accuracy

logging.debug('In network file.')

class Network():
    
    logging.debug('In network calss.')
    
    def __init__(self, para_choice = None):
        self.accuracy = 0
        self.mac_op = 0
        self.para_choice = para_choice
        self.network = {}
        self.kernel = []
        self.layers = []
        self.mac = 0
        self.mem = 0
        
    def create_random(self):
        "Create a random network."        
        for key in self.para_choice:
            self.network[key] = random.choice(self.para_choice[key])
        #logging.info(f'Network: {self.network}')

            
    def create_set(self, network):
        "Set network properties."
        
        logging.debug('In create set in network.')
        self.network = network
        
    def train(self,generation_no, network_no, time):
        "Train the network and record the accuracy."
        logging.debug('In Network, train function.')        
        #main.print_networks(self.network)
        
        if self.accuracy == 0 and self.mac_op == 0 and self.kernel == [] and self.layers == [] and self.mac == 0 and self.mem == 0:
            self.accuracy, self.mac_op, self.kernel, self.layers, self.mac, self.mem = train_and_accuracy(self.network, generation_no, network_no, time)
            #self.accuracy = train_and_accuracy(self.network,generation_no, network_no)

    def get_network(self):
        return self.network
    
    def print_network(self):
        "Print a network."
        logging.info('This is with Max_pool as first two layers.')
        logging.info(f'Network Info: {self.network}, Accuracy: {self.accuracy}, Mac and Memory: {self.mac_op}, Kernel_list: {self.kernel}, Layers: {self.layers}, Mac (Max allowed 8): {self.mac}, Mem (Max_allowed 2): {self.mem}')
        #logging.info(f"Network fitness: {(self.accuracy + self.mac_op)*100:.2f}")



