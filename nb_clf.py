#!/usr/bin/env python
# encoding: utf-8
#
# author: shenzhun
# date: 11/2013
#

import cProfile
import cPickle
import os
import random
import re
import sys
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
def get_words_list(dataset):
	'''
	Loading dataset and read contents, use tokenize to get tokens and lemmatize the words.
	'''

	# join the path and file name together
        spam_path = 'data/enron/pre/'+ dataset + '/spam/'
	ham_path = 'data/enron/pre/'+ dataset + '/ham/'
        spam_npl = [i[-1] for i in os.walk(spam_path)][0]
        ham_npl = [i[-1] for i in os.walk(ham_path)][0]

        spam_fl = (open(os.path.join(spam_path, j)).read().lower() for j in spam_npl)
	ham_fl = (open(os.path.join(ham_path, j)).read().lower() for j in ham_npl)

        splitter = re.compile("\\W*")
	english_stops = set(stopwords.words('english'))
	lemmatizer = WordNetLemmatizer()

	# tokenize the files into words
	spam_wl = [None]*len(spam_npl)
	for i,f in enumerate(spam_fl):
		spam_wl[i] = [word for word in (lemmatizer.lemmatize(w) for w in splitter.split(f) \
				if w not in english_stops and w.isalpha()) if len(word) > 2 and len(word) < 20]
        
	ham_wl = [None]*len(ham_npl)
	for i,f in enumerate(ham_fl):
		ham_wl[i] = [word for word in (lemmatizer.lemmatize(w) for w in splitter.split(f) \
				if w not in english_stops and w.isalpha()) if len(word) > 2 and len(word) < 20]

	return spam_wl, ham_wl


# create vocabulary list of these datasets
def get_feature_dict(words_list):
        '''
	draft vocabulary dict.
	'''
#        return list(set([w for words in words_list for w in set(words)]))
	return [i for i in Counter((w for words in words_list for w in words))]


# create vector for each file in these datasets
def get_files_vec(vocab_list, sample):
	'''
	translate files into vector
	'''

	sample_vec = [None]*len(sample)
	for i,f in enumerate(sample):
		word_freq = Counter(f[0])
		sample_vec[i] = [word_freq[j] for j in vocab_list]

	return sample_vec, [f[1] for f in sample]


# train n0aive bayes classifier using train matrix and train class labels
def train_NB(train_vec, train_spam_div):
        '''
	training naive bayes clssifier, get the vec parameters
	'''

	spam_num = train_vec[:train_spam_div].sum(axis=0) + train_spam_div
	ham_num = train_vec[train_spam_div:].sum(axis=0) + (len(train_vec)-train_spam_div)

	spam_denom = spam_num.sum() + len(train_vec[0])
	ham_denom = ham_num.sum() + len(train_vec[0])

	return log(spam_num/float(spam_denom)), log(ham_num/float(ham_denom))


# using trained classifier to classify the test sample
def classify_NB(test_vec, test_class, spam_lh, ham_lh, p_abusive):
	'''
	classify the test files using the classifier
	'''
	clf_diff = array([((i*spam_lh).sum()+log(p_abusive)- (i*ham_lh).sum()-log(1-p_abusive)) for i in test_vec])
	clf_class = (clf_diff > 0).astype(int)
	
#	for i in xrange(len(test_class)):
#		spam_p = sum(test_vec[i]*spam_lh) + log(p_abusive)
#		ham_p = sum(test_vec[i]*ham_lh) + log(1-p_abusive)
#		
#		if spam_p > ham_p:
#			clf_class[i] = 1
#			if test_class[i] == 1:
#				cnt_true_spam += 1
        cnt_true_spam = (clf_class*test_class).sum()
	clf_precision = float(cnt_true_spam)/clf_class.sum()
	clf_recall = float(cnt_true_spam)/test_class.sum()
        f_score = 2*clf_precision*clf_recall/(clf_precision+clf_recall)

	print 'precision = ', clf_precision
	print 'recall = ', clf_recall
	print 'f_score = ', f_score

# test the accuarcy of the classifer 
def test_NB(ds_name='enron1'):
	'''
	test naive bayes
	'''

	start = time()
	ratio = 0.7
	spam, ham = get_words_list(ds_name)
	random.shuffle(spam)
	random.shuffle(ham)

        train_spam_div = int(ratio*len(spam))
	train_ham_div = int(ratio*len(ham))

	train_set = [(i, 1) for i in spam[:train_spam_div]] + [(j, 0) for j in ham[:train_ham_div]]
	test_set = [(i, 1) for i in spam[train_spam_div:]] + [(j, 0) for j in ham[train_ham_div:]]
	
	words_list = spam[:train_spam_div] + ham[:train_ham_div]

	vocab_list = get_feature_dict(words_list)
	test_vocab_list = get_feature_dict(spam[train_spam_div:]+ham[train_ham_div:])

	train_vec, train_class = get_files_vec(vocab_list, array(train_set))

	# use chi-square feature selection method to select important features
	observed, expected = feature_selection.chi2(train_vec, train_class)
        chi_deviation = [0]*len(observed)
	for i in xrange(len(observed)):
		if expected[i] == 0 and expected[i] != observed[i]:
			chi_deviation[i] = 1000
		else:
			chi_deviation[i] = float((observed[i]-expected[i])**2)/expected[i]
	

	updated_vocab_list = [i[1] for i in sorted(zip(chi_deviation, vocab_list), reverse=True)][:2000]

	updated_train_vec, train_class = get_files_vec(updated_vocab_list, array(train_set))

	updated_test_vec, test_class = get_files_vec(updated_vocab_list, array(test_set))

	spam_vec, ham_vec = train_NB(array(updated_train_vec), array(train_spam_div))
        
	p_abusive = float(train_spam_div)/(train_spam_div+train_ham_div)

	classify_NB(array(updated_test_vec), array(test_class), array(spam_vec), array(ham_vec), p_abusive)
        print 'feature_lose = ', 1 - float(len(vocab_list))/len(set(vocab_list+test_vocab_list))	
	print time() - start, 'seconds'

if __name__ == '__main__':
        enron_set = ['enron1', 'enron2', 'enron3', 'enron4', 'enron5','enron6']
	if len(sys.argv) == 2 and sys.argv[1] in enron_set:
		cProfile.run('test_NB(sys.argv[1])', 'nb_clf_1.pyprof')
