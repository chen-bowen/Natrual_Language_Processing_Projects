import string
import random
import numpy as np


class Cipher:
    def __init__(self):
        self.build_cipher_mapping()

    def build_cipher_mapping(self):
        """ builds the subsitution cipher mapping for 26 letters """
        random.seed(7)
        letters_original = list(string.ascii_lowercase)
        letters_shuffled = random.sample(
            list(string.ascii_lowercase), len(list(string.ascii_lowercase))
        )
        self.cipher_mapping = {
            letters_original[i]: letters_shuffled[i] for i in range(len(letters_original))
        }
