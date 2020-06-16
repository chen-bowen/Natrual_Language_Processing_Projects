from sklearn.base import BaseEstimator
from sentiment_analysis.data.review_processor import ReviewProcessor
import pandas as pd
import numpy as np


class WordFrequencyVectorizer(BaseEstimator):
    """ Generate features in word frequency vectors """

    def __init__(
        self,
        categories=["electronics", "dvd", "kitchen_&_housewares", "books"],
        option="all",
    ):
        pass

    def set_word_index_mapping(self):
        processed_review = ReviewProcessor(self.categories, self.option)
        self.word_to_index_map = processed_review.word_to_index_map
        self.vocab_size = processed_review.vocab_size

    @staticmethod
    def get_word_frequency_vector(tokenized_review, word_to_index_map):
        """ Get the word frequency vector for one tokenized review"""
        # get count for all words appeared in the tokenized review
        word_frequency_count = Counter(tokenized_review)

        # map all the words to indices using the word to index map
        word_frequency_count_ind = {
            word_to_index_map[word]: count for word, count in word_frequency_count.items()
        }

        # flatten to a vector that equals to the vocabulary size
        word_frequency_vector = np.zeros(len(word_to_index_map))
        for ind, count in word_frequency_count_ind.items():
            word_frequency_vector[ind] += count

        return word_frequency_vector

    def fit(self, X, y=None):
        return self

    def word_frequency_matrix(self, X, y=None):
        """ Get the word frequency vectors for all tokenized reviews """
        word_frequency_vectors = list(
            map(
                partial(
                    self.get_word_frequency_vector,
                    word_to_index_map=self.word_to_index_map,
                ),
                X,
            )
        )

        word_frequency_matrix = np.stack(word_frequency_vectors, axis=0)
        return word_frequency_matrix
