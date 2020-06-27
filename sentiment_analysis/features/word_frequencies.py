from sklearn.base import BaseEstimator
from sentiment_analysis.data.review_processor import ReviewProcessor
from sentiment_analysis.utils.word_tokenizer import WordTokenizer
import pandas as pd
import numpy as np


class WordFrequencyVectorizer(BaseEstimator):
    """ Generate features in word frequency vectors, used for pipeline """

    def __init__(
        self,
        categories=["electronics", "dvd", "kitchen_&_housewares", "books"],
        option="all",
    ):
        self.tokenizer = WordTokenizer()
        self.set_word_index_mapping()

    def set_word_index_mapping(self):
        processed_review = ReviewProcessor(self.categories, self.option)
        self.word_to_index_map = processed_review.word_to_index_map
        self.vocab_size = processed_review.vocab_size

    @staticmethod
    def get_word_frequency_vector(review_text, word_to_index_map):
        """ Get the word frequency vector for one tokenized review"""
        # get the tokenized review from the review text
        tokenized_review = self.tokenizer.tokenize_sentence(review_text)

        # get count for all words appeared in the tokenized review
        word_frequency_count = Counter(tokenized_review)

        # map all the words to indices using the word to index map
        word_frequency_count_ind = {
            word_to_index_map[
                word if word in list(word_to_index_map.keys()) else "unknown_word"
            ]: count
            for word, count in word_frequency_count.items()
        }

        # flatten to a vector that equals to the vocabulary size + 1 (last index for unknown word)
        word_frequency_vector = np.zeros(len(word_to_index_map) + 1)
        for ind, count in word_frequency_count_ind.items():
            word_frequency_vector[ind] += count

        return word_frequency_vector

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
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
