#!/bin/python

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

all_files = open('all_words', 'r').read()
sw = stopwords.words('english')
words = nltk.word_tokenize(all_files.lower())
clean_words = [PorterStemmer(w) for w in words if w not in sw]
fd = nltk.probability.FreqDist(clean_words)
for i in fd:
	print i, fd[i]


