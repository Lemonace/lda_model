#encoding=utf-8
import gensim
from util import *
lda_model = gensim.models.LdaModel.load("./model/model.lda")
id2word = gensim.corpora.Dictionary.load("./model/model.lda.id2word")

#主题-词分布
topic_matrix = lda_model.get_topics()


#文章-主题分布
file_path = "/Users/lemonace/python/hlm/seged_data.txt"
no_below = 2
no_above = 0.5
word_count_dict, bag_of_words_corpus = get_corpus_train(file_path, no_below, no_above)
print type(bag_of_words_corpus)
lda_model.get_document_topics(bag_of_words_corpus)

