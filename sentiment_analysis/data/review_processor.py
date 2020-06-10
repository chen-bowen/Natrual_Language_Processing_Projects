import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import os
from collections import defaultdict, Counter
from itertools import chain
from functools import partial
import json


class ReviewProcessor:
    """ generate processed reviews and word index mapping """

    def __init__(
        self,
        categories=["electronics", "dvd", "kitchen_&_housewares", "books"],
        option="all",
    ):
        self.categories = categories
        self.option = option
        self._init_file_dir = os.path.dirname(__file__)
        self.build()

    def __load_reviews(self, cached_path):
        """ Load all reviews from the data folder """

        self.reviews = defaultdict(dict)
        np.random.seed(7)
        # populate reviews dict
        for review_type in ["positive", "negative"]:
            for cat in self.categories:
                file_path = os.path.join(
                    self._init_file_dir, "reviews/{}/{}.review".format(cat, review_type)
                )
                reviews_raw = BeautifulSoup(
                    open(file_path).read(), features="html.parser"
                )
                self.reviews[review_type][cat] = [
                    review.text for review in reviews_raw.find_all("review_text")
                ]
                np.random.shuffle(self.reviews[review_type][cat])
            # merge all categories into one if option is set to "all"
            if self.option == "all":
                self.reviews[review_type] = list(
                    chain(*list(self.reviews[review_type].values()))
                )
                np.random.shuffle(self.reviews[review_type])

        # save tokenized reviews to cache to speedup build process
        with open(cached_path, "w") as fp:
            json.dump(self.reviews, fp)

    @staticmethod
    def custom_tokenizer(s):
        """ build customer tokenizer by lower case, lemmatize and remove stopwords """
        wordnet_lemmatizer = WordNetLemmatizer()
        # lower case
        s = s.lower()
        # split string into words (tokens)
        tokens = nltk.tokenize.word_tokenize(s)
        # remove short words, they're probably not useful
        tokens = [t for t in tokens if len(t) > 2]
        # put words into base form
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
        # remove stopwords
        tokens = [t for t in tokens if t not in stopwords.words("english")]
        return tokens

    def __tokenize_reviews(self, cached_path):
        """" Tokenize all reviews, preprocess the reviews using custom tokenizer """
        self.reviews_tokenized = defaultdict(dict)
        for review_type in ["positive", "negative"]:
            if self.option != "all":
                for cat in self.categories:
                    self.reviews_tokenized[review_type][cat] = [
                        self.custom_tokenizer(i) for i in self.reviews[review_type][cat]
                    ]
            else:
                self.reviews_tokenized[review_type] = [
                    self.custom_tokenizer(i) for i in self.reviews[review_type]
                ]

        # save tokenized reviews to cache to speedup build process
        with open(cached_path, "w") as fp:
            json.dump(self.reviews_tokenized, fp)

    def build(self):
        """ Tokenize and build the word to index mapping, word to vector mapping"""
        # load reviews
        cached_path_reviews = os.path.join(self._init_file_dir, "cache/reviews.json")

        # use cached file if exists
        if os.path.exists(cached_path_reviews):
            with open(cached_path_reviews, "r") as fp:
                self.reviews = json.load(fp)

        else:
            print("Loading reviews ...")
            self.__load_reviews(cached_path_reviews)
            print("Completed")
            print("-----------------")

        # tokenize reviews
        cached_path_tokenized = os.path.join(
            self._init_file_dir, "cache/reviews_tokenized.json"
        )

        # use cached file if exists
        if os.path.exists(cached_path_tokenized):
            with open(cached_path_tokenized, "r") as fp:
                self.reviews_tokenized = json.load(fp)
        else:
            print("Tokenizing reviews ...")
            self.__tokenize_reviews(cached_path_tokenized)
            print("Completed")
            print("-----------------")

        # build word to index mapping, which is later used to map the word frequency column index to words
        all_unique_words = list(
            set(
                list(chain(*self.reviews_tokenized["positive"]))
                + list(chain(*self.reviews_tokenized["negative"]))
            )
        )

        self.word_to_index_map = {word: i for i, word in enumerate(all_unique_words)}
        self.vocab_size = len(self.word_to_index_map)

    @staticmethod
    def get_word_frequency_vector(tokenized_review, word_to_index_map, review_type):
        """ Get the word frequency vector for one tokenized review"""
        # get count for all words appeared in the tokenized review
        word_frequency_count = Counter(tokenized_review)

        # map all the words to indices using the word to index map
        word_frequency_count_ind = {
            word_to_index_map[word]: count for word, count in word_frequency_count.items()
        }

        # flatten to a vector that equals to the vocabulary size
        word_frequency_vector = np.zeros(len(word_to_index_map) + 1)
        for ind, count in word_frequency_count_ind.items():
            word_frequency_vector[ind] += count

        # add label to the end of the word frequency vector
        word_frequency_vector[-1] = 1 if review_type == "positive" else 0

        return word_frequency_vector

    @property
    def word_frequency_matrix(self):
        """ Get the word frequency vectors for all tokenized reviews """
        word_frequency_vectors = defaultdict(list)
        for review_type in ["positive", "negative"]:
            word_frequency_vectors[review_type] = list(
                map(
                    partial(
                        self.get_word_frequency_vector,
                        word_to_index_map=self.word_to_index_map,
                        review_type=review_type,
                    ),
                    self.reviews_tokenized[review_type],
                )
            )

        word_frequency_matrix = np.stack(
            list(chain(*list(word_frequency_vectors.values()))), axis=0
        )
        return word_frequency_matrix
