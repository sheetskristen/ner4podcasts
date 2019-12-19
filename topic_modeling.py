import csv
import pickle

import gensim
from gensim import corpora
import pyLDAvis.gensim
import nltk
from nltk.corpus import wordnet as wn


NUM_TOPICS = 3
STOPWORDS = set(nltk.corpus.stopwords.words('english'))


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def load_file(file):
    entities = []
    with open(file) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        next(tsvreader)
        for line in tsvreader:
            entity = line[0].split()
            normalized_ent = [get_lemma(word.lower()) for word in entity if word not in STOPWORDS]
            entities.append(''.join(normalized_ent))
            # entities.append(line[0].replace(" ", "_"))
            # entities.append(line[0])
    return entities

def gensim_model(text_data):
    text_data = [d.split() for d in text_data]
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')

    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
    # ldamodel.save('model_prototype.gensim')

    lsimodel = gensim.models.lsimodel.LsiModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary)
    # ldamodel.save('model_prototype.gensim')

    topics = lsimodel.print_topics()
    for topic in topics:
        print(topic)

def visualization():
    dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
    corpus = pickle.load(open('corpus.pkl', 'rb'))
    lda = gensim.models.ldamodel.LdaModel.load('model_prototype.gensim')
    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.display(lda_display)

def main():
    predicted = './topic_modeling_tsvs/predicted_topics.tsv'
    gold = './topic_modeling_tsvs/gold_topics.tsv'
    named_mentions = load_file(predicted)
    gold_mentions = load_file(gold)
    named_mentions.extend(gold_mentions)
    # print(named_mentions)
    gensim_model(named_mentions)
    # visualization()

if __name__ == "__main__":
    main()



