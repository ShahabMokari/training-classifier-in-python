#!/bin/env python


import os
import cPickle
import random
import re
from collections import Counter
import cProfile
from time import time
from operator import itemgetter

from numpy import ones
from numpy import log
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer


# obtain spam and ham files in the data directory, then tokenize the file into word without punctuations.
def obtain_filelist():
	# choose the datasets number
        corpus_no = abs(int(raw_input('Enter the number (1-5) to select corpus in enron(1, 2, 3, 4, 5): ')))
	while corpus_no == 0 or corpus_no > 5:
		corpus_no = abs(int(raw_input('Please re-enter the numver of corpora(1-5): ')))
	enron_corpus = 'enron' + str(corpus_no) 
        
	# join the path and file name together
        path = os.path.join('data/enron/pre/', enron_corpus)
        spam_path = os.path.join(path, 'spam')
        ham_path = os.path.join(path, 'ham')
        spam_dir = os.listdir(spam_path)
        ham_dir = os.listdir(ham_path)
        
	# get the filelist of the spam and ham datasets
        spam_filelist= [os.path.join(spam_path, f) for f in spam_dir if f.split('.')[-2] == 'spam']
        ham_filelist = [os.path.join(ham_path, f) for f in ham_dir if f.split('.')[-2] == 'ham']
        
	# tokenize the files into words
	spam_word_list = []
	ham_word_list = []
        
	tokenizer = RegexpTokenizer("[\w']+")
	english_stops = set(stopwords.words('english'))
	lemmatizer = WordNetLemmatizer()

	for i in spam_filelist:
		file = open(i).read()
#		file = re.sub(r"\d+", " ", open(i).read())
		words = [lemmatizer.lemmatize(word) for word in tokenizer.tokenize(file.lower()) if word not in english_stops]
		spam_word_list.append(words)
	
	for j in ham_filelist:
		file = open(j).read()
#		file = re.sub(r"\d+", " ", open(j).read())
		words = [lemmatizer.lemmatize(word) for word in tokenizer.tokenize(file.lower()) if word not in english_stops]
		ham_word_list.append(words)
	
	return spam_word_list, ham_word_list


# create vocabulary list of these datasets
def create_vocabularylist(train_set):
	
	spam_list = []
	ham_list = []
	for i in train_set:
		if i[1] == 1:
			spam_list.extend(i[0])
		else:
			ham_list.extend(i[0])
        word_freq = Counter(spam_list+ham_list)
	spam_dict = Counter(spam_list)
	ham_dict = Counter(ham_list)
 	vocab = [ i for i in word_freq if word_freq[i] > 10]

	return spam_dict, ham_dict, vocab


# create vector for each file in these datasets
def create_file2vec(vocab_list, sample):
	sample_vec = [[]]*len(sample)
	cnt = 0
	for file in sample:
		file_vec = [0]*len(vocab_list)
		for word in file[0]:
			if word in vocab_list:
				file_vec[vocab_list.index(word)] += 1
		sample_vec[cnt] = file_vec
		cnt += 1

	return sample_vec, [i[1] for i in sample]


# train naive bayes classifier using train matrix and train class labels
def train_NB(train_vec, train_class):

        # creating a 1 x num_words matrix using numpy 
	spam_num = ham_num = ones(len(train_vec[0]))
	spam_denom = ham_denom = 2

	for i in range(len(train_class)):
		if train_class[i] == 1:
			spam_num += train_vec[i]
			spam_denom += sum(train_vec[i])
		else:
			ham_num += train_vec[i]
			ham_denom += sum(train_vec[i])
	
	spam_lh = log(spam_num/spam_denom)
	ham_lh = log(ham_num/ham_denom)
	
	return spam_lh, ham_lh


# using trained classifier to classify the test sample
def classify_NB(vec2classify, spam_lh, ham_lh):
	spam_p = sum(vec2classify * spam_lh)
	ham_p = sum(vec2classify * ham_lh)

	if spam_p > ham_p:
		return 1
	else:
		return 0

	
# test the accuarcy of the classifer 
def test_NB():
	start = time()
	ratio = 2.0/3
	spam, ham = obtain_filelist()
	random.shuffle(spam)
	random.shuffle(ham)

        train_spam_div = int(ratio*len(spam))
	train_ham_div = int(ratio*len(ham))

	train_set = [(i, 1) for i in spam[:train_spam_div]] + [(j, 0) for j in ham[:train_ham_div]]
	test_set = [(i, 1) for i in spam[train_spam_div:]] + [(j, 0) for j in ham[train_ham_div:]]
	
	spam_dict, ham_dict, vocab_list = create_vocabularylist(train_set)
	train_vec, train_class = create_file2vec(vocab_list, train_set)
	test_vec, test_class = create_file2vec(vocab_list, test_set)

	spam_vec, ham_vec = train_NB(train_vec, train_class)

	count = 0
	for i in range(len(test_class)):
		cl_class = classify_NB(test_vec[i], spam_vec, ham_vec)
		if cl_class == test_class[i]:
			count += 1
	print float(count)/len(test_class)

	words_ratio = {}
	for i in range(len(vocab_list)):
		words_ratio[vocab_list[i]] = int(spam_vec[i]/ham_vec[i])
	print sorted(words_ratio.iteritems(), key=itemgetter(1), reverse=True)

	print time() - start

if __name__ == '__main__':
	cProfile.run('test_NB()', 'log_file.pyprof')

