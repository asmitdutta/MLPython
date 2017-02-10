#!/usr/bin/python
from nltk.tokenize import PunktSentenceTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize

input = raw_input()
tokenized = sent_tokenize(input)
for s in tokenized:
    print s