training-bayes-spam-filter
============================

## Background

This project is using Python and NLTK to build a spam filter with Naive Bayes Classifier with Enron Email Corpus. (http://www.cs.cmu.edu/~enron/); Python is an easy to learn, powerful programming language. It has efficient high-level data structures and a simple but effective approach to object-oriented programming. Python’s elegant syntax and dynamic typing, together with its interpreted nature, make it an ideal language for scripting and rapid application development in many areas on most platforms.

Natural Language Toolkit was developed in conjunction with a Computational Linguistics course at the University of Pennsylvania in 2001. It is a collection of modules and corpora, released under an open-source license, which allows students to learn and conduct research in NLP. NLTK can be used not only as a training complex, but also as a ready analytical tool or basis for the development of applied text processing systems. Nowadays it is widely used in linguistics, artificial intelligence, machine learning projects, etc.

Useful information can be found http://www.aueb.gr/users/ion/docs/ceas2006_paper.pdf .

## Project Details

1. Load enron email corpus into lists

2. Extracte word features from the emails

3. Making (features, label) list

4. Training and evaluating Naive Bayes Classifier

## Accuracy Of N-gram Testing

1-gram
------------------------
dataset  spam    ham
1        98.95%  93.54%
2        94.27%  99.51%
3        74.48%  99.82%
4        99.75%  90.67%
5        99.30%  99.71%
         ------  ------
         93.35%  96.65%

2-gram
------------------------
1        95.24%  97.20%
2        95.32%  98.79%
3        81.71%  99.79%
4        98.60%  99.33%
5        98.56%  99.81%
         ------  ------
         93.87%  98.98%

3-gram
------------------------
1        98.56%  99.81%
2        94.47%  97.84%
3        88.48%  98.18%
4        97.65%  99.62%
5        98.33%  99.14%
         ------  ------
         95.50%  98.92%

4-gram
------------------------
1        94.95%  92.22%
2        95.80%  94.96%
3        94.67%  94.02%
4        97.46%  99.33%
5        98.25%  98.67%
         ------  ------
         96.23%  95.84%
         



