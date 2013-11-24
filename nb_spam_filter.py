#!/bin/env python

import cProfile
import cPickle
import os
import random
import re
from collections import Counter
from time import time
from operator import itemgetter

from numpy import ones
from numpy import log
from numpy import array

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn import feature_selection


# obtain spam and ham files in the data directory, then tokenize the file into word without punctuations.
def get_words_list():
	'''
	Loading dataset and read contents, use tokenize to get tokens and lemmatize the words.
	'''

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
        spam_filelist= (os.path.join(spam_path, f) for f in spam_dir if f.split('.')[-2] == 'spam')
        ham_filelist = (os.path.join(ham_path, f) for f in ham_dir if f.split('.')[-2] == 'ham')
        
	# tokenize the files into words
	spam_word_list = []
	ham_word_list = []
        
#	tokenizer = RegexpTokenizer("[\w']+")
        splitter = re.compile("\\W*")
	english_stops = set(stopwords.words('english'))
	lemmatizer = WordNetLemmatizer()

	for i in spam_filelist:
		f = open(i).read()
		split_words = (lemmatizer.lemmatize(w) for w in splitter.split(f.lower()))
		words = [ w for w in split_words if w not in english_stops and len(w) > 2 and len(w) < 20 and w.isalpha()]
		spam_word_list.append(words)
	
	for j in ham_filelist:
		f = open(j).read()
		split_words = (lemmatizer.lemmatize(w) for w in splitter.split(f.lower()))
		words = [ w for w in split_words if w not in english_stops and len(w) > 1 and len(w) < 20 and w.isalpha()]
		ham_word_list.append(words)

	return spam_word_list, ham_word_list


# create vocabulary list of these datasets
def get_feature_dict(words_list):
        '''
	draft vocabulary dict.
	'''

	word_freq = Counter((w for words in words_list for w in words))
        vocab = [i for i in word_freq if word_freq[i] > 0]

	return vocab


# create vector for each file in these datasets
def get_files_vec(vocab_list, sample):
	'''
	translate files into vector
	'''

	sample_vec = []
	for f in sample:
		file_vec = [Counter(f[0])[i] for i in vocab_list]
	        sample_vec.append(file_vec)

	return sample_vec, [f[1] for f in sample]


# train naive bayes classifier using train matrix and train class labels
def train_NB(train_vec, train_spam_div):
        '''
	training naive bayes clssifier, get the vec parameters
	'''

        # creating a 1 x num_words matrix using numpy 
#	spam_num = ham_num = ones(len(train_vec[0]))
#	spam_denom = ham_denom = k_smoothing

	spam_num = train_vec[:train_spam_div].sum(axis(1)) + ones(len(train_vec[0]))
	ham_num = train_vec[train_spam_div:].sum(axis(1)) + ones(len(train_vec[0]))

	spam_denom = spam_num.sum() + len(train_vec[0])
	ham_denom = ham_num.sum() + len(train_vec[0])

#	for i in xrange(len(train_class)):
#		if train_class[i] == 1:
#			spam_num += train_vec[i]
#			spam_denom += sum(train_vec[i])
#		else:
#			ham_num += train_vec[i]
#			ham_denom += sum(train_vec[i])
#	
#	spam_lh = log(spam_num/float(spam_denom))
#	ham_lh = log(ham_num/float(ham_denom))
	
	return log(spam_num/float(spam_denom)), log(ham_num/float(ham_denom))


# using trained classifier to classify the test sample
def classify_NB(test_vec, test_class, spam_lh, ham_lh, p_abusive):
	'''
	classify the test files using the classifier
	'''
	cnt_true_spam = 0
	clf_class = [0]*len(test_class)
	for i in xrange(len(test_class)):
		spam_p = sum(test_vec[i]*spam_lh) + log(p_abusive)
		ham_p = sum(test_vec[i]*ham_lh) + log(1-p_abusive)
		
		if spam_p >= ham_p and test_class[i] == 1:
			cnt_true_spam += 1
			clf_class[i] = 1

	clf_precision = float(cnt_true_spam)/clf_class.count(1)
	clf_recall = float(cnt_true_spam)/test_class.count(1)
        f_score = 2*clf_precision*clf_recall/(clf_precision+clf_recall)

	print 'precision = ', clf_precision
	print 'recall = ', clf_recall
	print 'f_score = ', f_score

# test the accuarcy of the classifer 
def test_NB():
	'''
	test naive bayes
	'''

	start = time()
	ratio = 0.7
	spam, ham = get_words_list()
	random.shuffle(spam)
	random.shuffle(ham)

        train_spam_div = int(ratio*len(spam))
	train_ham_div = int(ratio*len(ham))

	train_set = [(i, 1) for i in spam[:train_spam_div]] + [(j, 0) for j in ham[:train_ham_div]]
	test_set = [(i, 1) for i in spam[train_spam_div:]] + [(j, 0) for j in ham[train_ham_div:]]
	
	words_list = spam[:train_spam_div] + ham[:train_ham_div]

	vocab_list = get_feature_dict(words_list)

	train_vec, train_class = get_files_vec(vocab_list, array(train_set))

	# use chi-square feature selection method to select important features
	observed, expected = feature_selection.chi2(train_vec, train_class)
        chi_deviation = [0]*len(observed)
	for i in xrange(len(observed)):
		if expected[i] == 0 and expected[i] != observed[i]:
			chi_deviation[i] = 1000
		else:
			chi_deviation[i] = float((observed[i]-expected[i])**2)/expected[i]
	

	updated_vocab_list = [i[1] for i in sorted(zip(chi_deviation, vocab_list), reverse=True)][:1000]

	updated_train_vec, train_class = get_files_vec(updated_vocab_list, array(train_set))

	updated_test_vec, test_class = get_files_vec(updated_vocab_list, array(test_set))

	spam_vec, ham_vec = train_NB(array(updated_train_vec), train_spam_div)
        
	p_abusive = float(train_spam_div)/(train_spam_div+train_ham_div)

	classify_NB(updated_test_vec, test_class, spam_vec, ham_vec, p_abusive)
	
	print time() - start, 'seconds'

if __name__ == '__main__':
	cProfile.run('test_NB()', 'running_log.pyprof')

