#!/bin/python
import nltk
import os
from numpy import ones, zeros
import operator
import cPickle


def obtain_filelist():
        corpus_no = abs(int(raw_input('Enter the number (1-5) to select corpus in enron(1, 2, 3, 4, 5): ')))
	while corpus_no == 0 or corpus_no > 5:
		corpus_no = abs(int(raw_input('Please re-enter the numver of corpora(1-5): ')))
	enron_corpus = 'enron' + str(corpus_no) 

	ng = abs(int(raw_input('Enter the degree of n-gram(1-4 is suggested): ')))
	while ng == 0 or ng > 4:
                ng = abs(int(raw_input('Please re-enter the degree (1-4): ')))
	
        path = os.path.join('data/enron/pre/', enron_corpus)
        spam_path = os.path.join(path, 'spam')
        ham_path = os.path.join(path, 'ham')
        spam_dir = os.listdir(spam_path)
        ham_dir = os.listdir(ham_path)

        spam_filelist= [os.path.join(spam_path, f) for f in spam_dir]
        ham_filelist = [os.path.join(ham_path, f) for f in ham_dir]

	spam_word_list = []
	ham_word_list = []

	for i in spam_filelist:
		file = open(i).read()
		words = nltk.tokenize.regexp_tokenize(file.lower(), "[\w']+")
		spam_word_list.append(words)
	
	for j in ham_filelist:
		file = open(j).read()
		words = nltk.tokenize.regexp_tokenize(file.lower(), "[\w']+")
		ham_word_list.append(words)

	return spam_word_list, ham_word_list


def create_vocabularylist(words_list, num=21):
	freq_dist = {}
	for list in words_list:
		for word in list:
			if word in freq_dist.keys():
				freq_dist[word] += 1
			else:
				freq_dist.setdefault(word, 1)
	word_freq = sorted(freq_dist.iteritems(), key=operator.itemgetter(1))
	set_feat = [word[0] for word in word_freq[-num:-1]]

	return set_feat


def create_file2vec(vocab_list, all_file_words, feat_class):
	doc_vector = [0]*len(vocab_list)
	for file in all_file_words:
		for word in file:
			if word in vocab_list:
				doc_vector[vocab_list.index(word)] += 1
	return doc_vector, feat_class

def train_NB(train_mat, train_class):
	doc_num_train = len(train_mat)
	num_words = len(train_mat[0])
	spam_num = zeros(num_words)
	ham_num = zeros(num_words)

	spam_denom = 0
	ham_denom = 0

	for i in range(doc_num_train):
		if train_class[i] == 1:
			spam_num += train_mat[i]
			spam_denom += sum(train_mat[i])
		else:
			ham_num += train_mat[i]
			ham_denom += sum(train_mat[i])
	
	spam_vect = spam_num/spam_denom
	ham_vect = ham_num/ham_denom
	
	return spam_vect, ham_vect


def classify_NB(vec2classify, spam_vect, ham_vect):
	spam_prt = sum(vec2classify * spam_vect)
	ham_prt = sum(vec2classify * ham_vect)

	if spam_prt > ham_prt:
		return 1
	else:
		return 0


def test_NB():
	spam, ham = obtain_filelist()
	train_sample = spam[:1000] + ham[:1000]
	train_class = [1]*1000 + [0]*1000

	test_sample = spam[1000:] + ham[1000:]
	test_class = [1]*(len(spam)-1000)+[0]*(len(ham) - 1000)

#	vocab_list = create_vocabularylist(spam+ham)
        with open('sorted_words_freq.pickle', 'rb') as f:
		vocab_list = cPickle.load(f)
	tr_mat, tr_class = create_file2vec(vocab_list, train_sample, train_class)
	ts_mat, ts_class = create_file2vec(vocab_list, test_sample, test_class)

	spam_vec, ham_vec = train_NB(train_mat, train_class)

	count = 0
	for i in range(len(ts_mat)):
		cl_class = classify_NB(ts_mat[i], spam_vec, ham_vec)
		if cl_class == ts_class[i]:
			count += 1
	print 'accuracy is %f' % float(count)/len(ts_mat)
