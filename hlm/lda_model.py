from util import *

file_path = "/Users/lemonace/python/hlm/seged_data.txt"
no_below = 2
no_above = 0.99
num_topics = 15
extend_stopwords = "/Users/lemonace/python/hlm/stopwords"
word_count_dict, bag_of_words_corpus = get_corpus_train(file_path, extend_stopwords, no_below, no_above)
lda_model = gensim.models.LdaModel(bag_of_words_corpus, num_topics=num_topics, id2word=word_count_dict, iterations=2000)
for corpus in bag_of_words_corpus:
    print len(corpus)
    print lda_model.get_document_topics(bow=corpus)
lda_model.save("./model/model1.lda")
for i in xrange(0, 15):
    print lda_model.print_topic(i, 20)