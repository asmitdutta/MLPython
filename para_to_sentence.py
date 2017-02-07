#!/usr/bin/python

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import wordnet

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

synonyms = []
antonyms = []

syns = wordnet.synsets("program")
syns2 = wordnet.synsets("plan")
print(syns.wup_similarity)
print(syns[0].lemmas())
print(syns[0].lemmas()[0].name())
print(syns[0].definition())
print(syns[0].examples())

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))


for i in tokenized[5:]:
    words = nltk.word_tokenize(i)
    tagged = nltk.pos_tag(words)
    namedEnt = nltk.ne_chunk(tagged)
    namedEnt.draw()