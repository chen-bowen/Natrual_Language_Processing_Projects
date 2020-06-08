import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import os
from collections import defaultdict
from itertools import chain


class ReviewProcessor:
    """ generate processed reviews and word index mapping """

    def __init__(
        self,
        categories=["electronics", "dvd", "kitchen_&_housewares", "books"],
        option="all",
    ):
        self.categories = categories
        self.option = option
        self.build()

    def __load_reviews(self):
        """ Load all reviews from the data folder """
        _init_file_dir = os.path.dirname(__file__)
        self.reviews = defaultdict(dict)
        np.random.seed(7)
        # populate reviews dict
        for review_type in ["positive", "negative"]:
            for cat in self.categories:
                file_path = os.path.join(
                    _init_file_dir, "reviews/{}/{}.review".format(cat, review_type)
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

    @staticmethod
    def customer_tokenizer(s):
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

    def build(self):
        """ Tokenize and build the word to index mapping, word to vector mapping"""
        # load reviews
        print("Loading reviews ...")
        self.__load_reviews()
        print("Completed")
        print("-----------------")

        # tokenize reviews
        print("Tokenizing Reivews ...")
        self.reviews_tokenized = defaultdict(dict)
        for review_type in ["positive", "negative"]:
            if self.option != "all":
                for cat in self.categories:
                    self.reviews_tokenized[review_type][cat] = [
                        self.customer_tokenizer(i) for i in self.reviews[review_type][cat]
                    ]
            else:
                self.reviews_tokenized[review_type] = [
                    self.customer_tokenizer(i) for i in self.reviews[review_type]
                ]
        print("Completed")
        print("-----------------")

        # build word to index mapping, which is later used to map the word frequency column index to words
        print("Building word index mapping ...")
        all_unique_words = list(
            set(
                list(chain(*self.reviews_tokenized["positive"]))
                + list(chain(*self.reviews_tokenized["negative"]))
            )
        )

        self.word_to_index_map = {word: i for i, word in enumerate(all_unique_words)}
        print("Completed")
        print("-----------------")
