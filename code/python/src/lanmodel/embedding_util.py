from distutils.util import strtobool

from fasttext import load_model
from os import listdir

import datetime

import multiprocessing

import gc

import gensim
from gensim.models import Word2Vec


class Corpus(object):
    """
    supervised org/per pair classifier

    """
    input_folder = None
    start_file_index = None

    def __init__(self, input_folder, start_file_index):
        self.input_folder = input_folder
        self.start_file_index=start_file_index

    def __iter__(self):
        files = [f for f in listdir(self.input_folder)]
        files=sorted(files)

        index=-1
        for f in files:
            index+=1
            if index<self.start_file_index:
                continue
            print(str(datetime.datetime.now())+" started from file:"+f)
            for line in open(self.input_folder+"/"+f,
                             encoding='utf-8', errors='ignore'):
                # assume there's one document per line, tokens separated by whitespace
                yield line.lower().split()


'''
cbow_or_skip: 1 means skipgram, otherwise cbow
'''
def train_word2vec(input_folder, start_file_index, out_model_file, cbow_or_skip):
    cores = multiprocessing.cpu_count()
    print('num of cores is %s' % cores)
    gc.collect()

    sentences = Corpus(input_folder,start_file_index)
    print(str(datetime.datetime.now()) + ' training started...')
    model = Word2Vec(sentences=sentences,
                     size=300, window=10, min_count=5, sample=1e-4, negative=5, workers=cores, sg=cbow_or_skip)

    print(str(datetime.datetime.now()) + ' training completed, saving...')
    model.save(out_model_file)
    print(str(datetime.datetime.now()) + ' saving completed')


def load_emb_model(embedding_format:str, embedding_file:str):
    if embedding_format == 'gensim':
        print("\tgensim format")
        emb_model = gensim.models.KeyedVectors.load(embedding_file, mmap='r')
    elif embedding_format == 'fasttext':
        print("\tfasttext format")
        emb_model = load_model(embedding_file)
    else:
        binary = embedding_format.split("=")[1]
        print("\tword2vec format, binary="+str(strtobool(binary)))
        emb_model = gensim.models.KeyedVectors. \
            load_word2vec_format(embedding_file, binary=strtobool(binary))
    return emb_model

def embedding_to_text_format(embedding_file:str, out_file:str):
    model=load_emb_model('gensim', embedding_file)
    model.wv.save_word2vec_format(out_file, binary=False)

if __name__ == "__main__":
    #train_word2vec(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
    embedding_to_text_format("/home/zz/Work/data/embeddings/wop/w2v_desc_skip.bin",
                             "/home/zz/Work/data/embeddings/wop/w2v_desc_skip.txt")