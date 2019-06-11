from genericpath import isfile
from os import listdir

import datetime

import multiprocessing

import gc

import sys
from gensim.models import Doc2Vec, Word2Vec


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


if __name__ == "__main__":
    train_word2vec(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))