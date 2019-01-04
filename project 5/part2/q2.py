import json
import matplotlib.pyplot as plt
import numpy as np
import datetime, time
import pytz
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn import svm, linear_model
from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, recall_score, precision_score
import itertools
from sklearn.neural_network import MLPClassifier

class MyTokenizer(object):
    def __init__(self):
        self.wnl = SnowballStemmer("english")
    def __call__(self, doc):
        doc = doc.translate({ord(c): None for c in string.punctuation})
        return [self.wnl.stem(t) for t in word_tokenize(doc)]

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        From SKLearn Documentation
        """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# 0 - WA, 1 - MA, 2 - others
def checkLocation(locationStr):
    WashingtonLocations = ['Seattle', 'Washington', 'WA', 'Kirkland']
    MassachusettsLocations = ['Massachusetts', 'Boston', 'MA', 'Springfield', 'Mass.']
    WA_black_list = ["dc", "d.c.", "d.c."]
    for subStr in WashingtonLocations:
        if subStr in locationStr:
            for subStr2 in WA_black_list:
                if subStr2 not in locationStr:
                    return 0
    for subStr in MassachusettsLocations:
        if subStr in locationStr:
            return 1
    return 2

pst_tz = pytz.timezone('US/Pacific')
df = pd.DataFrame()

titles = []
locations = []
locationStrs = []

input_file = open('/Volumes/DATA/Documents/Term 10/219 Project5/tweet_data/tweets_#superbowl.txt')
for line in input_file:
    data = json.loads(line)

    locationCode = checkLocation(data['tweet']['user']['location'])
    if (locationCode != 2):
        titles.append("".join(b for b in data['title'] if ord(b) < 128))
        locations.append(locationCode)
        locationStrs.append(data['tweet']['user']['location'])

input_file.close()

df = pd.DataFrame({'titles':titles,
                   'locations':locations,
                   })

X_train, X_test, y_train, y_test = train_test_split(df['titles'], df['locations'], test_size=0.1, random_state=42)

vectorizer = CountVectorizer(stop_words=text.ENGLISH_STOP_WORDS, tokenizer=MyTokenizer(), min_df=2)
X_train_counts = vectorizer.fit_transform(X_train.values.astype(str))
X_test_counts = vectorizer.transform(X_test.values.astype(str))
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

DimensionRedcutionModel = TruncatedSVD(n_components=50)
W_train = DimensionRedcutionModel.fit_transform(X_train_tfidf)
W_test = DimensionRedcutionModel.transform(X_test_tfidf)

# uncomment one of the classifier to select which one to use
# clf = svm.LinearSVC()
# clf = MLPClassifier()
clf = linear_model.LogisticRegression()

y_score = clf.fit(W_train, y_train).decision_function(W_test) #Use this line for SVM and Logistic Classifiers
y_predicted = clf.fit(W_train, y_train).predict(W_test) #Use this line for SVM and Logistic Classifiers


# y_score = clf.fit(W_train, y_train).predict_proba(W_test)[:,1] # Use this line for MLP with SVD
# y_predicted = clf.fit(W_train, y_train).predict(W_test)# Use this line for MLP with SVD

# y_score = clf.fit(X_train_tfidf, y_train).predict_proba(X_test_tfidf)[:,1]  # Use this line for MLP without SVD
# y_predicted = clf.fit(X_train_tfidf, y_train).predict(X_test_tfidf)  # Use this line for MLP without SVD

# Plot ROC curve
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=[12,10]).set_tight_layout(True)
lw = 2
plt.subplot(221)
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# Plot confusion matrix
cnf_matrix = confusion_matrix(y_test, y_predicted)
np.set_printoptions(precision=2)
plt.subplot(222)
plot_confusion_matrix(cnf_matrix, classes=['Washington', 'Massachusetts'],
                      title='Confusion matrix, without normalization')
plt.subplot(223)
plot_confusion_matrix(cnf_matrix, classes=['Washington', 'Massachusetts'], normalize=True,
                      title='Normalized confusion matrix')

# Annotation
plt.subplot(224)
plt.text(0.1,0.1,'df = {}\nDimension-reduction:{}\n'
         'classifier:{}\naccuracy_score = {:0.4f}\n'
         'recall_score = {:0.4f}\n'
         'precision_score = {:0.4f}'.format(vectorizer.min_df,
                                            DimensionRedcutionModel.__class__.__name__,
                                            clf.__class__.__name__,
                                            accuracy_score(y_test,y_predicted),
                                            recall_score(y_test,y_predicted),
                                            precision_score(y_test,y_predicted)))