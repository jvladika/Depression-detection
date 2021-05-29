import numpy as np
import pandas as pd
import csv
import timeit
import datetime
import collections
import torch
import data


class Vocab:
    def __init__(self, directory, max_size = -1, min_freq = 1):
        assert max_size > 2 or max_size == -1

        extracted = data.extract(directory)

        with open(directory) as file:
            words= [item for instance in file for item in instance.split(', ')[0].split()]
        counter = collections.Counter(words)
        min_filtered = [(item, counter[item]) for item in counter if counter[item] >= min_freq]
        sorted_words = sorted(min_filtered, key=lambda x: x[1], reverse=True)

        self.stoi = {'<PAD>': 0, '<UNK>': 1}
        index: int = 2
        for stem in sorted_words:
            self.stoi[stem[0]] = index
            index += 1
            if index == max_size:
                break

    def encode(self, stems):
        result = torch.empty(len(stems), dtype=torch.long).fill_(self.stoi['<UNK>'])
        for i, stem in enumerate(stems):
            if stem in self.stoi:
                result[i] = self.stoi[stem]

        return result

    def __getitem__(self, item):
        return self.stoi[item]

    def __len__(self):
        return len(self.stoi)


class WordVectorLoader:

    def __init__(self, embed_dim):
        self.embed_index = {}
        self.embed_dim = embed_dim

    def load_glove(self, file_name):
        df = pd.read_csv(file_name, header=None, sep=' ', encoding='utf-8', quoting=csv.QUOTE_NONE)
        for index, row in df.iterrows():
            word = row[0]
            coefs = np.asarray(row[1:], dtype='float32')
            self.embed_index[word] = coefs
        try:
            self.embed_dim = len(coefs)
        except:
            pass

    def create_embedding_matrix(self, embeddings_file_name, word_to_index, max_idx, sep=' ', init='zeros', print_each=10000, verbatim=False):
        # Initialize embeddings matrix to handle unknown words
        if init == 'zeros':
            embed_mat = np.zeros((max_idx + 1, self.embed_dim))
        elif init == 'random':
            embed_mat = np.random.rand(max_idx + 1, self.embed_dim)
        else:
            raise Exception('Unknown method to initialize embeddings matrix')

        start = timeit.default_timer()
        with open(embeddings_file_name) as infile:
            for idx, line in enumerate(infile):
                elem = line.split(sep)
                word = elem[0]

                if verbatim is True:
                    if idx % print_each == 0:
                        print('[{}] {} lines processed'.format(datetime.timedelta(seconds=int(timeit.default_timer() - start)), idx), end='\r')

                if word not in word_to_index:
                    continue

                word_idx = word_to_index[word]

                if word_idx <= max_idx:
                    embed_mat[word_idx] = np.asarray(elem[1:], dtype='float32')


        if verbatim == True:
            print()

        return embed_mat

    def generate_embedding_matrix(self, word_to_index, max_idx, init='zeros'):
        # Initialize embeddings matrix to handle unknown words
        if init == 'zeros':
            embed_mat = np.zeros((max_idx + 1, self.embed_dim))
        elif init == 'random':
            embed_mat = np.random.rand(max_idx + 1, self.embed_dim)
        else:
            raise Exception('Unknown method to initialize embeddings matrix')

        for word, i in word_to_index.items():
            if i > max_idx:
                continue
            embed_vec = self.embed_index.get(word)
            if embed_vec is not None:
                embed_mat[i] = embed_vec

        return embed_mat

    def generate_centroid_embedding(self, word_list, avg=False):
        centroid_embedding = np.zeros((self.embed_dim, ))
        num_words = 0
        for word in word_list:
            if word in self.embed_index:
                num_words += 1
                centroid_embedding += self.embed_index.get(word)
        # Average embedding if needed
        if avg is True:
            if num_words > 0:
                centroid_embedding /= num_words
        return centroid_embedding




if __name__ == '__main__':

    word_vector_loader = WordVectorLoader()

    word_vector_loader.create_embedding_matrix('/home/vdw/data/dumps/glove/glove.6B.100d.txt', verbatim=True)