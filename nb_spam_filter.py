#!/bin/python
import os
import cPickle
import random
from collections import Counter
import cProfile
from time import time

import nltk
from numpy import ones, zeros
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
	all_words = []
        
        try:
		spam_word_list = cPickle.load(open(enron_corpus+'_spam_word_list.pkl', 'r'))
		ham_word_list = cPickle.load(open(enron_corpus+'_ham_word_list.pkl', 'r'))
		all_words = cPickle.load(open(enron_corpus+'_all_words.pkl', 'r'))
	except:
		tokenizer = RegexpTokenizer("[\w']+")
		english_stops = set(stopwords.words('english'))
		lemmatizer = WordNetLemmatizer()

        	for i in spam_filelist:
        		file = open(i).read()
        		words = [lemmatizer.lemmatize(word) for word in tokenizer.tokenize(file.lower()) if word not in english_stops]
        		spam_word_list.append(words)
        		all_words.extend(words)
        
                with open(enron_corpus+'_spam_word_list.pkl', 'w') as f:
		        cPickle.dump(spam_word_list, f)
        	
		for j in ham_filelist:
        		file = open(j).read()
        		words = [ lemmatizer.lemmatize(word) for word in tokenizer.tokenize(file.lower()) if word not in english_stops]
        		ham_word_list.append(words)
        		all_words.extend(words)

		with open(enron_corpus+'_ham_word_list.pkl', 'w') as f:
			cPickle.dump(ham_word_list, f)

	        with open(enron_corpus+'_all_words.pkl', 'w') as f:
			cPickle.dump(all_words, f)

	return spam_word_list, ham_word_list, all_words


# create vocabulary list of these datasets
def create_vocabularylist(words_list, num=1, dataset_no='enron1'):
	
	with open(dataset_no+'_spam_word_list.pkl', 'r') as f:
		spam_word_list = cPickle.load(f)
	
	with open(dataset_no+'_ham_word_list.pkl', 'r') as f:
		ham_word_list = cPickle.load(f)

	spam_set = set()
	spam_list = []
	for list in spam_word_list:
		spam_list.extend(list)
		spam_set = spam_set | set(list)

	ham_set = set()
	ham_list = []
	for list in ham_word_list:
		ham_list.extend(list)
		ham_set = ham_set | set(list)
        
	spam_dict = Counter(spam_list)
	ham_dict = Counter(ham_list)

	set_common = ham_set & spam_set
	
#       stemmer = PorterStemmer()        
#	clean_words= [stemmer.stem(w) for w in words_list if (len(w) > 1) and (len(w) <= 20)]
#	word_freq = nltk.probability.FreqDist(clean_words)
        
	clean_words = [ w for w in words_list if (len(w) > 1) and (len(w) <= 20)]
        word_freq = Counter(clean_words)
 	set_feat = [ i for i in word_freq if word_freq[i] > num]

	return set_feat, word_freq, spam_dict, ham_dict, set_common


# create vector for each file in these datasets
def create_file2vec(vocab_list, all_file_words, feat_class):
	all_vector = [[]]*len(all_file_words)
	cnt = 0
	for file in all_file_words:
		doc_vector = [0]*len(vocab_list)
		stemmer = PorterStemmer()
		for word in file:
			stem_word = stemmer.stem(word)
			if stem_word in vocab_list:
				doc_vector[vocab_list.index(stem_word)] += 1
		all_vector[cnt] = doc_vector
		cnt += 1

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
	start = time()
	ratio = 2.0/3
	spam, ham, all_words = obtain_filelist()
	random.shuffle(spam)
	random.shuffle(ham)

        train_spam_div = int(ratio*len(spam))
	train_ham_div = int(ratio*len(ham))

	train_sample = spam[:train_spam_div] + ham[:train_ham_div]
	train_class = [1]*train_spam_div + [0]*train_ham_div

        test_sample = spam[train_spam_div:] + ham[train_ham_div:]
	test_class = [1]*(len(spam)-train_spam_div)+[0]*(len(ham) - train_ham_div)
	
	vocab_list, words_freq, spam_dict, ham_dict, set_common = create_vocabularylist(all_words)
	tr_mat, tr_class = create_file2vec(list(set_common), train_sample, train_class)
	ts_mat, ts_class = create_file2vec(list(set_common), test_sample, test_class)

	spam_vec, ham_vec = train_NB(tr_mat, tr_class)
        

	count = 0
	for i in range(len(ts_mat)):
		cl_class = classify_NB(ts_mat[i], spam_vec, ham_vec)
		if cl_class == ts_class[i]:
			count += 1
	print float(count)/len(ts_mat)

	words_ratio = {}
	for i in range(len(list(set_common))):
		words_ratio[list(set_common)[i]] = int(spam_vec[i]/ham_vec[i])
	end = time() - start
	print sorted(words_ratio.iteritems(), key=itemgetter(1), reverse=True)[:10]
	print end

if __name__ == '__main__':
	cProfile.run('test_NB()', 'log_file.pyprof')

