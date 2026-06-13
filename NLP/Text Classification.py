from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.stem.snowball import SnowballStemmer



# Extracting features from text files
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

print("\n".join(twenty_train.data[5].split("\n")[:3])) #prints first line of the first  data file
print((twenty_train.data[5].split("\n")[:3])) #prints first line of the first data file

#A.getnnz( axis=1 ) gets the number of non-zeros per row, A.getnnz( axis=0 ) per column, and A.getnnz() == A.nnz
X_train_counts.getnnz(axis=0)

# TF-IDF term frequecy inverse document frequency
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Training Naive Bayes (NB) classifier on training data
clf1 = MultinomialNB()
clf1.fit(X_train_tfidf, twenty_train.target)

# Performance of NB Classifier
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)

#count_vect2 = CountVectorizer()
X_test_counts = count_vect.transform(twenty_test.data)

#tfidf_transformer2 = TfidfTransformer()
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

pred=clf1.predict(X_test_tfidf)

np.mean(pred == twenty_test.target)

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)


predicted = text_clf.predict(twenty_test.data)
np.mean(predicted == twenty_test.target)

predicted
twenty_train.target_names[predicted[3]]
predicted


# Training Support Vector Machines - SVM and calculating its performance
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42))])

text_clf_svm = text_clf_svm.fit(twenty_train.data, twenty_train.target)
predicted_svm = text_clf_svm.predict(twenty_test.data)
np.mean(predicted_svm == twenty_test.target)

parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)


# To see the best mean score and the params, run the following code
gs_clf.best_score_
gs_clf.best_params_

parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}

gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(twenty_train.data, twenty_train.target)


gs_clf_svm.best_score_
gs_clf_svm.best_params_


# NLTK
# Removing stop words
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])


# Stemming Code
nltk.download()

stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
                             ('mnb', MultinomialNB(fit_prior=False))])

text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)

predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)

np.mean(predicted_mnb_stemmed == twenty_test.target)