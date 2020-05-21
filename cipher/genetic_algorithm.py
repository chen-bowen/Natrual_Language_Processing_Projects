import numpy as np
import random
import string


class GeneticAlgorithm:

    POOL_SIZE = 20
    OFFSPRING_POOL_SIZE = 10

    def __init__(self):
        self.__initialize_dna_pool()

    def __initialize_dna_pool(self):
        """ Generates a DNA pool """
        # get all 26 letters
        all_letters = list(string.ascii_lowercase)
        # generate 20 random permutation of the all_letters as the dna_pool
        self.dna_pool = [
            "".join(list(np.random.permutation(all_letters)))
            for _ in range(self.POOL_SIZE)
        ]
        self.offspring_pool = []

    @staticmethod
    def random_swap(sequence):
        """ random swap two characters at random positions """
        # generate random positions
        index_1, index_2 = random.sample(list(np.arange(len(sequence))), 2)
        # swap positions
        seq_list = list(sequence)
        seq_list[index_1], seq_list[index_2] = seq_list[index_2], seq_list[index_1]
        return "".join(seq_list)

    def evolve_offspring(self):
        """ evolves offspring by random swaps for every dna sequence in the dna pool """
        for dna in self.dna_pool:
            # evolve 10 offsprings for each dna in the dna pool
            self.offspring_pool += [
                self.random_swap(dna) for _ in range(self.OFFSPRING_POOL_SIZE)
            ]
        return self.offspring_pool + self.dna_pool
