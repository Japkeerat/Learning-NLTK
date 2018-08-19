# Learning-NLTK

This repository contains source codes to some small applications created while learning natural language processing in Python using NLTK(Natural Language Toolkit).

Dependencies
---
<ul>
  <li>SkLearn</li>
  <li>Scipy</li>
  <li>nltk (quite obvious)</li>
</ul>

<b>CustomTokenizer.py</b> file contains the code to make a custom tokenizer using PaktSentenceTokenizer which is an unsupervised machine learning model.

<b>PartOfSpeechTagging.py</b> file contains code that is responsible for tagging nouns in the sentences with some predefined classes.

<b>NaiveBayesForMovieReviews.py</b> makes a supervised machine learning model where it uses movie reviews database for learning to classify reviews into positive and negative reviews. This method is highly volatile in accuracy and varies from 60% to 90% for every test run without changing any parameters.

<b>SkLearnNaiveBayes.py</b> uses some more supervised machine learning models present in SkLearn library and combines it with NLTK. Accuracy of models varies from 60% to 75%.

<b>EnsemblingModels.py</b> makes a custom model using all the models made in SkLearnNaiveBayes.py and runs tests on the custom model. Averagely, this gives accuracy of 70%.
