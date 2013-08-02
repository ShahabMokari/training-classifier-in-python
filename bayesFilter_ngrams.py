import os
import re
from nltk.corpus import stopwords
from nltk import NaiveBayesClassifier
import nltk.classify
from nltk.tokenize import wordpunct_tokenize
from collections import defaultdict
import string

# create features dictionary
def indicate_word(message):
	"""Create a dictionary of entries{word: True} for every unique word in each message.
	"""
        features = defaultdict(list)
	for k in message:
		features[k] = True
	return features

# make (feature, label) lists
def label_features(filelist, label):
	""" Make (features, label) list, features are the dictionaries { word: True} of each
	message, label is 'spam' or 'ham'. Thus all the features are listed of each message'
	"""
	features_labels = []
	for fl in filelist:
		tokens = []
		for sent  in nltk.sent_tokenize(open(fl).read()):
			for word in nltk.word_tokenize(sent):
				if word not in string.punctuation:
					tokens.append(word)
		ngrams_tokens = nltk.ngrams(tokens, 2)
	        features = indicate_word(ngrams_tokens)
	        features_labels.append((features, label))
        return features_labels

# train and evaluate naive bayes classifier
def evaluate_classifier(train_set, test_spam, test_ham):
	""" Using NaiveBayesClassifier.train() method from NLTK to train the train_set (spam + ham),
	then classifier is used to evaluate the accuracy of test Spam, Ham. Finally, the most informative 
	features are showed.
	"""
	classifier = NaiveBayesClassifier.train(train_set)
	print('Test Spam accuracy: {0:.2f} %'.format(100 * nltk.classify.accuracy(classifier, test_spam)))
	print('Test Ham accuracy: {0:.2f} %'.format(100 * nltk.classify.accuracy(classifier, test_ham)))
        print classifier.show_most_informative_features(20)


# fetch corpora from enron emails into list of files
def main():
        enron_corpus = raw_input('Enter the corpora of enron(1, 2, 3, 4, 5): ')
        path = os.path.join('data/enron/pre/', enron_corpus)
        spam_path = os.path.join(path, 'spam')
        ham_path = os.path.join(path, 'ham')

        spam_dir = os.listdir(spam_path)
        ham_dir = os.listdir(ham_path)

        spam_filelist= [os.path.join(spam_path, f) for f in spam_dir]
        ham_filelist = [os.path.join(ham_path, f) for f in ham_dir]

        spam_size = len(spam_filelist)
        ham_size = len(ham_filelist)


        # divide the emails into two parts, train set and test set
        train_spam_filelist = spam_filelist[:int(spam_size*0.3)]
        train_ham_filelist = ham_filelist[:int(spam_size*0.3)]

        test_spam_filelist = spam_filelist[int(spam_size*0.3):]
        test_ham_filelist = ham_filelist[int(ham_size*0.3):]


        # label train and test data sets
        train_spam = label_features(train_spam_filelist, 'spam')
        train_ham = label_features(train_ham_filelist, 'ham')
        train_set = train_spam + train_ham
        test_spam = label_features(test_spam_filelist, 'spam')
        test_ham = label_features(test_ham_filelist, 'ham')

        # evaluate the Naive Bayes classifier with data sets
        evaluate_classifier(train_set, test_spam, test_ham)

if __name__ == '__main__':
        main()

