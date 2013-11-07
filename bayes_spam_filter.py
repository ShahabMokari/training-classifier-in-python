#!/bin/python
import nltk
import os
from numpy import ones, zeros
import operator


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


def create_vocabularylist(words_list, num=81):
	freq_dist = {}
	for list in words_list:
		for word in list:
			if word in freq_dist.keys():
				freq_dist[word] += 1
			else:
				freq_dist.setdefault(word, 1)
	word_freq = sorted(freq_dist.iteritems(), key=operator.itemgetter(1))
	set_feat = word_freq[-num:-1]

	return set_feat


def create_doc2Vec(vocab_list, doc_words):
	doc_vector = [0]*len(vocab_list)
	for word in doc_words:
		if word in vocab_list:
			doc_vector[vocab_list.index(word)] += 1
		else:
			print "Not in the dataset fatures"
	
	return doc_vector


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
	train_sample = spam[:1000].extend(ham[:1000])
	train_class = [1]*2000

	test_sample = spam[1000:].extend(ham[1000:])
	test_class = [0]*(len(spam)+len(ham) - 2000)

	all_filelist = [i for i in spam.extend(ham)]
	vocab = [create_vocabularylist(nltk.word_tokenize(open(file).read())) for file in all_filelist]

	train_mat = []
	for file in open([i for i in train_sample]):	
		train_mat.append(create_doc2vec(vocab, nltk.word_tokenize(file)))

	test_mat = []
	for file in open([i for i in test_sample]):
		test_mat.append(create_doc2vec(vocab, nltk.word_tokenize(file)))
	
	spam_vec, ham_vec = train_NB(train_mat, train_class)
