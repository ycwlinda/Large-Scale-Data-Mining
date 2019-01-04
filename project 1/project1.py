
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer
import string
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize  
'''
Qusetion a
'''


categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers','footers','quotes'), shuffle=True, random_state=42)
all = fetch_20newsgroups(subset='all', categories=categories, remove=('headers','footers','quotes'), shuffle=True, random_state=42)

plt.hist(train.target, 15)
plt.title("the number of training documents per class")
x= np.arange(8)
plt.xticks(x,categories,rotation=90, fontsize=10)
#plt.show()



'''
Question b
'''
stop_words = text.ENGLISH_STOP_WORDS

stemmer = SnowballStemmer("english")
EXTRA = "'-- - ''abcdefghijklmnopqrstuvwxyz0123456789.`~!@#$%^&*()_=+{}[]\|:;< >} 's ... `` n't ?/"
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [i for i in tokens if i not in string.punctuation]
    tokens = [i for i in tokens if i not in EXTRA]
    stems = stem_tokens(tokens, stemmer)
    return stems

count_vect=CountVectorizer(min_df=2, max_df=0.7, tokenizer = tokenize, stop_words=stop_words)
train_counts=count_vect.fit_transform(train.data)
'''
print("extracted term number:", len(count_vect.get_feature_names()))
print(train_counts.shape)
tfidf_transform=TfidfTransformer()
train_tfidf=tfidf_transform.fit_transform(train_counts)
print(train_tfidf.shape)
'''

'''
Question c
'''
from sklearn.feature_extraction.text import TfidfVectorizer
TFxIDF = TfidfVectorizer(analyzer='word',tokenizer=tokenize, stop_words=stop_words)
#TFxIDF_data = TFxIDF.fit_transform(all.data)
train = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'), shuffle=True, random_state=42)


target_name2idx = {}
for idx, target_name in enumerate(train.target_names):
    target_name2idx[target_name] = idx

data_by_class = [""] * len(train.target_names)
for idx, data in enumerate(train.data):
    fname = train.filenames[idx]
    class_name = fname.split('/')[-2]
    class_idx = target_name2idx[class_name]

    data_by_class[class_idx] += data + "\n"

my_list=['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','misc.forsale','soc.religion.christian']

for category in my_list:
    categories = [category]
    sub_data = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42, remove=('headers','footers','quotes'))
    TFxIDF_sub = TfidfVectorizer(analyzer='word',tokenizer=tokenize, max_features=10, stop_words=stop_words)
    sub_count = TFxIDF_sub.fit_transform(sub_data.data)
    #print (TFxIDF_sub.vocabulary_.keys())
count_vect = text.CountVectorizer(min_df=1, stop_words=stop_words, tokenizer=tokenize)
Xc_train_counts = count_vect.fit_transform(data_by_class)

#--------------------------------

# TFxICF
tficf_transformer = text.TfidfTransformer()
X_train_tficf = tficf_transformer.fit_transform(Xc_train_counts)
print "Shape of TFxICF matrix: %s" % (X_train_tficf.shape,)

#--------------------------------

print ""
import operator
from operator import itemgetter

# Finding the top 10 in certain class
reverse_vocab_dict = {}
for term in count_vect.vocabulary_:
    term_idx = count_vect.vocabulary_[term]
    reverse_vocab_dict[term_idx] = term

target_classes = ['comp.sys.ibm.pc.hardware' , 'comp.sys.mac.hardware',  'soc.religion.christian', 'misc.forsale']
for class_name in target_classes:
    X_train_tficf_array = X_train_tficf.toarray()
    class_idx = target_name2idx[class_name]
    sig_arr = [(idx, val) for idx, val in enumerate(X_train_tficf_array[class_idx])]
    top10 = sorted(sig_arr, key=itemgetter(1), reverse=True)[:10]
    #print top10
    
    print "Top 10 significant terms in class %s:" % class_name
    for idx, val in enumerate(top10):
        term_idx, sig_val = val
        print "%-16s(significance = %f)" % (reverse_vocab_dict[term_idx], sig_val)

    print ""    # new line for every target class



