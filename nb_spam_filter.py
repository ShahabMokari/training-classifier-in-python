#!/bin/python
import os
import operator
import cPickle
import random
import nltk
from numpy import ones, zeros
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


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

	all_words = []
        
	for i in spam_filelist:
		file = open(i).read()
		words = nltk.word_tokenize(file.lower())
		spam_word_list.append(words)
		all_words.extend(words)

	for j in ham_filelist:
		file = open(j).read()
		words = nltk.word_tokenize(file.lower())
		ham_word_list.append(words)
		all_words.extend(words)

	return spam_word_list, ham_word_list, all_words


# create vocabulary list of these datasets
def create_vocabularylist(words_list, num=201):
	'''
	freq_dist = {}
	for list in words_list:
		for word in list:
			if word in freq_dist.keys():
				freq_dist[word] += 1
			else:
				freq_dist.setdefault(word, 1)
	
	word_freq = sorted(freq_dist.iteritems(), key=operator.itemgetter(1))
        
        # load cPickle file of word freq in order to save time
        with open('sorted_words_freq.pkl', 'rb') as f:
		word_freq = cPickle.load(f)
	'''
        stemmer = PorterStemmer()        
	stop_words = stopwords.words('english')
	clean_words= [stemmer.stem(w) for w in words_list if (w not in stop_words) and (len(w) > 1) and (len(w) <= 20)]
	word_freq = nltk.probability.FreqDist(clean_words)
	set_feat = [ i for i in word_freq if word_freq[i] > 1]

	return set_feat, word_freq


# create vector for each file in these datasets
def create_file2vec(vocab_list, all_file_words, feat_class):
	all_vector = []*len(all_file_words)
	for file in all_file_words:
		doc_vector = [0]*len(vocab_list)
		stemmer = PorterStemmer()
		for word in file:
			stem_word = stemmer.stem(word)
			if stem_word in vocab_list:
				doc_vector[vocab_list.index(stem_word)] += 1
		all_vector.append(doc_vector)
	return all_vector, feat_class


# train naive bayes classifier using train matrix and train class labels
def train_NB(train_mat, train_class):
	doc_num_train = len(train_mat)
	num_words = len(train_mat[0])
#	spam_num = zeros(num_words)
#	ham_num = zeros(num_words)

        # creating a 1 x num_words matrix using numpy 
	spam_num = ones(num_words)
	ham_num = ones(num_words)

	spam_denom = 2
	ham_denom = 2

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

# using trained classifier to classify the test sample
def classify_NB(vec2classify, spam_vect, ham_vect):
	spam_prt = sum(vec2classify * spam_vect)
	ham_prt = sum(vec2classify * ham_vect)

	if spam_prt > ham_prt:
		return 1
	else:
		return 0

# test the accuarcy of the classifer 
def test_NB():
	spam, ham, all_words = obtain_filelist()
	random.shuffle(spam)
	random.shuffle(ham)
	train_sample = spam[:1000] + ham[:1000]
	train_class = [1]*1000 + [0]*1000

        test_sample = spam[1000:] + ham[1000:]
	test_class = [1]*(len(spam)-1000)+[0]*(len(ham) - 1000)
	
	vocab_list = create_vocabularylist(all_words)
	tr_mat, tr_class = create_file2vec(vocab_list, train_sample, train_class)
	ts_mat, ts_class = create_file2vec(vocab_list, test_sample, test_class)

	spam_vec, ham_vec = train_NB(tr_mat, tr_class)

	count = 0
	for i in range(len(ts_mat)):
		cl_class = classify_NB(ts_mat[i], spam_vec, ham_vec)
		if cl_class == ts_class[i]:
			count += 1
	print float(count)/len(ts_mat)
