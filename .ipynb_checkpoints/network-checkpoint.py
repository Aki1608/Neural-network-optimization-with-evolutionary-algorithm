#! Python3

#import Main
import random
import logging
from train import train_and_accuracy

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
            
    def create_set(self, network):
        '''
        Set network properties.
        
        Args:
            network_no (int): in each generation there are fixed number of network which are created.
        
        '''
        
        logging.debug('In create set in network.')
        self.network = network
        
    def train(self,generation_no, network_no, time):
        '''Train the network and record the accuracy.
        
        Args:
            generation_no (int): number of generation (iteration for evolutionary algorithm).
            network_no (int): in each generation there are fixed number of network which are created.
            time: current time ("current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")")
         '''
        logging.debug('In Network, train function.')        
        
        if self.accuracy == 0 and self.mac_op == 0 and self.kernel == [] and self.layers == [] and self.mac == 0 and self.mem == 0:
            self.accuracy, self.mac_op, self.kernel, self.layers, self.mac, self.mem = train_and_accuracy(self.network, generation_no, network_no, time)

    def get_network(self):
        return self.network
    
    def print_network(self):
        "Print a network."
        logging.info('This is with Max_pool as first two layers.')
        logging.info(f'Network Info: {self.network}, Accuracy: {self.accuracy}, Mac and Memory: {self.mac_op}, Kernel_list: {self.kernel}, Layers: {self.layers}, Mac: {self.mac}, Mem: {self.mem}')
        #logging.info(f"Network fitness: {(self.accuracy + self.mac_op)*100:.2f}")



