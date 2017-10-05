#encoding=utf-8
from sklearn.decomposition import PCA
from sklearn.cluster import *
import pickle
import gensim
from util import *
lda_model = gensim.models.LdaModel.load("./model/model1.lda")

id2word = gensim.corpora.Dictionary.load("./model/model1.lda.id2word")

#主题-词分布
topic_matrix = lda_model.get_topics()
#文章-主题分布
file_path = "/Users/lemonace/python/hlm/seged_data.txt"
no_below = 2
no_above = 0.99
extend_stopwords = "/Users/lemonace/python/hlm/stopwords"
word_count_dict, bag_of_words_corpus = get_corpus_train(file_path, extend_stopwords, no_below, no_above)
data = []

for i in xrange(0, 15):
    print lda_model.print_topic(i, 20)
for corpus in bag_of_words_corpus:
    v = lda_model.get_document_topics(corpus)
    print len(corpus)
    print v
    tmp = [0] * 15
    for item in v:
        tmp[item[0]] = item[1]
    data.append(tmp)
print data
pca=PCA(n_components=3)

kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
print kmeans.labels_

newData=pca.fit_transform(data)


with open("/Users/lemonace/python/hlm/dd.txt", "wb") as output:
    pickle.dump(newData, output)
