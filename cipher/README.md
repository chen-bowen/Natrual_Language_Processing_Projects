# Cipher Decryption

When we want to send a private message that only ourselves understand, we will use a encryption map that subsitute letters with some other letters, which makes the original message unreadable. However, suppose we don't know the original encryption map, is there a way to estimate/approximate the encryption map using NLP ? In this project, we will use the bigram character model to build a likelihood evaluation metric for the estimated encryption map, and which will be produced by a really interesting technique named genetic algorithm. The whole project contains 3 modeules - `LanguageModel`, `Encoder` and `Genetic Algorithm`. Using the combination of these 3 modules, we are able to run an iterative approach to estimate the encryption map from a paragraph of text. 

### Language Model

The language model serves as the base evaluation metric as we randomize and refine the encryption map. When we encrypt using the true map and decrypt using the estimated map, the more correct-to-the-truth map will return a higher log likelihood score since the language model is trained over real English words that contains specific ordering of the alphabetical characters. The language model is trained using bigrams and Markov Property that calculates the probabiltities of every two character combination in English alphabets. The more frequent occurence combinations will have a higher probability score. The module is built in an automatically instantiated class that generates the probabilities scores of all combinations immediately. We could get the mapping the following way.

```python
from model.language_model import LanguageModel

lm = LanguageModel()
probabilities = lm.log_M

```
### Encoder

The encoder module generates the true character mapping for encryption and decryption. The mapping is stored as an attribute of the Encoder class in a python dict. When the `Encoder` class is instantiated and the `encode` method is called, the inputted original message will be preprocessed (removing all non-alphabetical characters) and subsituted on a character level using the true encryption map built in the class.

```python
from model.encoder import Encoder

original_message = """I then lounged down the street and found,
as I expected, that there was a mews in a lane which runs down
by one wall of the garden. I lent the ostlers a hand in rubbing
down their horses, and received in exchange twopence, a glass of
half-and-half, two fills of shag tobacco, and as much information
as I could desire about Miss Adler, to say nothing of half a dozen
other people in the neighbourhood in whom I was not in the least
interested, but whose biographies I was compelled to listen to.
"""
encoder = Encoder()
encoded_message = encoder.encode(original_message)

```
