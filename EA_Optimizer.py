#!/usr/bin/env python
# coding: utf-8

from network import Network
import random
import logging

logging.debug('In optimizer.')



class Optimizer():
    
    logging.debug('In Optimizer class.')
    
    def __init__(self,para_choice, retain = 0.4, 
                random_select = 0.2, mutate_chance = 0.4, all_networks = []):
        "Create an optimizer."
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.para_choice = para_choice
        self.all_networks = all_networks
        
    def create_population(self, count: int) -> list:
        
        logging.debug('In create population in optimizer.')
        
        '''
        Create a population of random networks.
        
        Args:
            count (int): Number of networks to generate,
            aka size of population.
            
        Returns: 
            list: Population of networks objects.
        '''
        
        population = []
        
        #No same network will be generated.
        for i in range(0, count):
            repeat = False
            network= Network(self.para_choice)
            network.create_random()
            self.all_networks.append(network.get_network)
            
            if len(self.all_networks) > 1:
                for i in range(len(self.all_networks)):
                    if network.get_network() == self.all_networks[i]:
                        self.all_networks.pop()
                        repeat = True
            while repeat == True:
                network= Network(self.para_choice)
                network.create_random()
                self.all_networks.append(network.get_network)
                for i in range(len(all_networks)):
                    if network.get_network() == self.all_networks[i]:
                        self.all_networks.pop()
                        repeat = True
                        break
                    else:
                        repeat = False
                        
            population.append(network)
        return population
    
    @staticmethod
    def fitness(network: list) -> float:
        '''
        Return the fitness value.
        
        Args:
            network (list): list of parameters of network.
            
        Returns:
            float: The fitness value
        '''
        logging.debug('In fitness in opt.')
        
        return (network.accuracy + (network.mac_op))
    
    def grade(self, population: int) -> float:
        """
        Find average fitness for a population.
        
        Args:
            population (list): The population of networks.
            
        Returns:
            float: The average accuracy of the popualtion
        
        """
        logging.debug('In grade function. in opt.')
        addition = 0
        for network in population:
            addition += self.fitness(network)
            
        return additon / float((len(population)))
    

    def mutate(self, network: list) -> list:
        """
        Randomly mutate one part of the network.
        
        Args:
            network (dict): The network parameter to mutate.
            
        Returns:
            Network: A randomly mtated network object.
        """
        logging.info('In mutate in opt.')
        
        mutation = random.choice(list(self.para_choice.keys()))
        
        network.network[mutation] = random.choice(self.para_choice[mutation])
        
        logging.info('Mutation done.')
        
        return network
    
    def breed(self, mother: dict, father: dict) -> list:
        """
        Make two children as parts of their parents.
        
        Args:
            Mother (dict): Network parameters
            Father (dict): Network parameters
            
        Reeturns:
            list: Two network object
        """
        
        logging.info('In breed function. in opt.')
        
        children = []
        for _ in range(2):
            
            child = {}
            
            for param in self.para_choice:
                child[param] = random.choice(
                    [mother.network[param], father.network[param]]
                )
                
            network = Network(self.para_choice)
            network.create_set(child)
            
            if self.mutate_chance > random.random():
                network = self.mutate(network)
                     
            children.append(network)
            logging.info('Children created.')
        return children
    
    
    def evolve(self, population: list) -> list:
        """
        Evolve a population of networks.
        
        Args:
            population (list): A list of network parameters.
            
        Returns:
            list: The evolved population of networks.
        """
        
        logging.debug('In evolve function in optimizer.')
        
        graded = [(self.fitness(network), network) for network in population]
        
        print(graded)
        
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)] 
        logging.info(f'lenght of graded in evolve (should be same as popuation): {len(graded)}')
        
        retain_length = int(len(graded)*self.retain)
        
        parents = graded[:retain_length]
        logging.info('Parents created in opt.')
        logging.info(f'length of parents: {len(parents)}')
        
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)
                
        parents_length = len(parents)
        desired_length = len(population) - parents_length
        logging.info(f'population length: {len(population)}')
        logging.info(f'Desired length (population - parents_length): {desired_length}')
        children = []
        
        logging.info('Creating children.')
        
        while len(children) < desired_length:
            male = random.randint(0, parents_length-1)
            female = random.randint(0, parents_length-1)
            
            logging.debug('After while. in evolve in opt.')
        
            if male != female:
                male = parents[male]
                female = parents[female]

                babies = self.breed(male, female)
                
                logging.debug('Creating babies. in evolve in opt.')
                logging.info(f'Lenght of babies: {len(babies)}')

                for baby in babies:
                    if len(children) < desired_length:
                        children.append(baby)
                        logging.info(f'Length of childern inside "baby loop" in EA_optimizer: {len(children)}')

                logging.info('Children created.')
            
        parents.extend(children)
        logging.debug('Parent and children in one list.')
        logging.info(f'Lenght of final population after parents and babies combined: {len(parents)}')
        
        return parents        