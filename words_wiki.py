
from __future__ import unicode_literals
import logging
import multiprocessing
import sys

from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


import numpy
import spacy
from spacy.language import Language


logger = logging.getLogger("logger")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
logging.root.setLevel(level=logging.INFO)
wiki = WikiCorpus(sys.argv[1], lemmatize=False, dictionary={});

limit = None
if len(sys.argv) > 2:
    limit=int(sys.argv[2])

i = 0
sentences_out_stream = open("sentences.txt", "w")
sentences_out_stream = open("sentences.txt", "w")

logger.info("building sentences list");
for text in wiki.get_texts():
    while (len(text) > 0):
        slice = text[0:9999]
        text = text[10000:]
        string = " ".join(slice) + "\n"
        sentences_out_stream.write(string)
    i = i + 1
    logger.info("%i articles slices processed", i)
    if limit != None and i > limit:
        break

sentences_out_stream.close()
logger.info("building vectors from sentences")
sentences_in_stream = open("sentences.txt","rb")
vectors = Word2Vec(LineSentence(sentences_in_stream), size=300,window=5, min_count=5,workers=multiprocessing.cpu_count())

logger.info("saving vectors to disk in word2vec format")
vectors.wv.save_word2vec_format("vectors.txt")
sentences_in_stream.close()

logger.info("encoding word2vec vectors in spacy format")
i = 1
nlp = spacy.load("it")
with open("vectors.txt", 'rb') as file_:
    header = file_.readline()
    nr_row, nr_dim = header.split()
    nlp.vocab.reset_vectors(width=int(nr_dim))
    for line in file_:
        line = line.rstrip().decode('utf8')
        pieces = line.rsplit(' ', int(nr_dim))
        word = pieces[0]
        vector = numpy.asarray([float(v) for v in pieces[1:]], dtype='f')
        nlp.vocab.set_vector(word, vector)  # add the vectors to the vocab
        logger.info("%i vectors encoded",i)
        i=i+1


logger.info("saving spacy model with vectors to disk")
nlp.to_disk("it-vectors-model")
logger.info("end")