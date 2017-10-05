from gensim.parsing.preprocessing import STOPWORDS
import gensim


stopwords = STOPWORDS

def tokenize(text, extend_stopwords):
    words = [w for w in text.split(" ") if w not in stopwords]
    extend_stopwords_list = []
    es =  open(extend_stopwords, "r")
    for line in es:
        extend_stopwords_list.append(line.strip())
    words = [w for w in words if w not in extend_stopwords_list]
    return words

def read_document(file_path):
    documents = []
    try:
        f = open(file_path)
        for line in f:
            documents.append(line.strip())
        return documents
    except:
        return []

def get_corpus_train(file_path, extend_stopwords, no_below, no_above):
    documents = read_document(file_path)
    processed_docs = [tokenize(doc, extend_stopwords) for doc in documents]
    for doc in processed_docs:
        print len(doc)
    #obtain: (word_id:word)
    word_count_dict = gensim.corpora.Dictionary(processed_docs)
    word_count_dict.filter_extremes(no_below=no_below, no_above=no_above)
    # word must appear >5 times, and no more than 20% documents
    bag_of_words_corpus = [word_count_dict.doc2bow(pdoc) for pdoc in processed_docs]
    return word_count_dict, bag_of_words_corpus

def load_id2word(model_path):
    return gensim.corpora.Dictionary.load(model_path)

def get_corpus_test(file_path, word_count_dict):
    documents = read_document(file_path)
    processed_docs = [tokenize(doc) for doc in documents]
    bag_of_words_corpus = [word_count_dict.doc2bow(pdoc) for pdoc in processed_docs]
    return bag_of_words_corpus