# Natrual Language Processing Mini Projects

This repo contains the natural language processing miniprojects from Lazy Programmer's NLP class

### Project 1 - Cipher Encryption-Decryption

When we want to send a private message that only ourselves understand, we will use a encryption map that subsitute letters with some other letters, which makes the original message unreadable. However, suppose we don't know the original encryption map, is there a way to estimate/approximate the encryption map using NLP ? In this project, we will use the bigram character model to build a likelihood evaluation metric for the estimated encryption map, and which will be produced by a really interesting technique named genetic algorithm. The whole project contains 3 modeules - LanguageModel, Encoder and Genetic Algorithm (specific implementation see the README inside the folder). Using the combination of these 3 modules, we are able to run an iterative approach to estimate the encryption map from a paragraph of text. 

Medium blog post regarding this topic [Understanding Bigrams: Cipher Decryption with Language Modeling and Genetic Algorithm](https://medium.com/@bowenchen/understanding-bigrams-cipher-decryption-with-language-modeling-and-genetic-algorithm-1283ee0b3b80)

## Project 2 - Spam Detector

The spam detector is frequently used in the industry to filter out the irritating spam messages inbox from both email and texts. This project will be a really simple exercise on the Kaggle SMS spam detection dataset with Naive Bayes classifier. 
