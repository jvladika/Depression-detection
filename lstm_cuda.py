import numpy as np
import pandas as pd
import csv
import timeit
import datetime
import collections
import torch
from pathlib import Path
import data
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from torch import nn
from dataclasses import dataclass
from sklearn import metrics
import itertools
import math
import random


class Vocab:
    def __init__(self, directory, max_size=-1, min_freq=1):
        assert max_size > 2 or max_size == -1
        extracted_positive = data.extract(directory + '/positive_examples_anonymous')
        extracted_negative = data.extract(directory + '/negative_examples_anonymous')
        extracted = {**extracted_positive, **extracted_negative}
        sentences = [item[1] for instance in extracted for item in extracted[instance]]

        stemmer = SnowballStemmer("english")
        words = []
        for i, sentence in enumerate(sentences):
            words += [stemmer.stem(x) for x in word_tokenize(sentence)]

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

    def set_word2vec(self, word2vec):
        self.embed_index = word2vec.wv

    def generate_embedding_matrix(self, vocab, max_elements=-1, init='zeros'):
        # Initialize embeddings matrix to handle unknown words
        if max_elements == -1:
            max_elements = len(vocab)
        if init == 'zeros':
            embed_mat = np.zeros((max_elements, self.embed_dim))
        elif init == 'random':
            embed_mat = np.random.rand(max_elements, self.embed_dim)
            embed_mat[vocab['<PAD>']] = np.zeros(self.embed_dim)
        else:
            raise Exception('Unknown method to initialize embeddings matrix')

        for word in vocab.stoi:
            index = vocab[word]
            if index >= max_elements:
                continue
            if word in self.embed_index:
                embed_mat[index] = self.embed_index[word]
        return nn.Embedding.from_pretrained(torch.tensor(embed_mat), freeze=True, padding_idx=vocab['<PAD>'])


class DatasetConcat(torch.utils.data.Dataset):
    def __init__(self, directory: str, vocab: Vocab):
        self.vocab = vocab

        extracted_positive = data.extract(directory + '/positive_examples_anonymous')
        extracted_negative = data.extract(directory + '/negative_examples_anonymous')
        users_positive = [(user, [post[1] for post in extracted_positive[user]]) for user in extracted_positive]
        users_negative = [(user, [post[1] for post in extracted_negative[user]]) for user in extracted_negative]

        stemmer = SnowballStemmer("english")

        self.instances = []
        for user, posts in users_positive:
            total_post = ' '.join(posts)
            stems_encoded = vocab.encode([stemmer.stem(x) for x in word_tokenize(total_post)])
            self.instances.append((user, stems_encoded, 1))

        for user, posts in users_negative:
            total_post = ' '.join(posts)
            stems_encoded = vocab.encode([stemmer.stem(x) for x in word_tokenize(total_post)])
            self.instances.append((user, stems_encoded, 0))

    def __len__(self):
        return len(self.instances)

    def get_full(self, item):
        return self.instances[item]

    def __getitem__(self, item):
        instance = self.instances[item]
        return instance[1], torch.tensor(instance[2])


class TF_IDF:
    def __init__(self, data, vocab: Vocab):
        self.data = data
        depressed_data = [x for x in self.data if x[2] == 1]

        stems = list(itertools.chain.from_iterable([x[1] for x in self.data]))
        stems = [x for x in stems if x in vocab.stoi]
        word_frequencies = collections.Counter(stems)

        d_stems = list(itertools.chain.from_iterable([x[1] for x in depressed_data]))
        d_stems = [x for x in d_stems if x in vocab.stoi]
        d_word_frequencies = collections.Counter(d_stems)

        max_d_freq = d_word_frequencies[max(d_word_frequencies, key=lambda x: d_word_frequencies[x])]
        self.tf_scores = {}
        for word in d_word_frequencies:
            self.tf_scores[word] = 0.5 + (0.5 * word_frequencies[word]) / max_d_freq

        documents = len(self.data)
        word_frequencies = {}
        for _, document, _ in self.data:
            for word in set(document):
                if word in word_frequencies:
                    word_frequencies[word] += 1
                else:
                    word_frequencies[word] = 1

        self.idf_scores = {}
        for word in word_frequencies:
            self.idf_scores[word] = math.log(documents / word_frequencies[word]) * math.tanh(
                math.pow((word_frequencies[word]) * 0.025, 3))

        tfidf_scores = {}
        tfidf_scores_words = {}

        self.min_tf_score = self.tf_scores[min(self.tf_scores, key=lambda x: self.tf_scores[x])]

        for word in self.idf_scores:
            if word in self.tf_scores:
                tfidf_scores[vocab[word]] = self.tf_scores[word] * self.idf_scores[word]
                tfidf_scores_words[word] = self.tf_scores[word] * self.idf_scores[word]
            else:
                try:
                    tfidf_scores[vocab[word]] = self.min_tf_score * self.idf_scores[word]
                    tfidf_scores_words[word] = self.min_tf_score * self.idf_scores[word]
                except:
                    pass
        self.scores = tfidf_scores
        self.word_dict = tfidf_scores_words
        self.min_tfidf_score = min([self.scores[x] for x in self.scores])

    def get(self, x):
        if x in self.scores:
            return self.scores[x]
        if x in self.idf_scores:
            self.scores[x] = self.min_tf_score * self.idf_scores[x]
            return self.scores[x]
        return self.min_tfidf_score

    def __call__(self, items):
        result = torch.empty(len(items))
        for i, item in enumerate(items):
            result[i] = self.get(item.item())
        return result

    def __getitem__(self, key):
        return self.get(key)


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.hidden_dim = config['hidden']

        self.lstm = nn.LSTM(config['embedding_dim'] + 1, config['hidden'], config['lstm_layers'], bidirectional=False)
        self.hidden = nn.Linear(config['hidden'], 1)
        self.sigmoid = nn.Sigmoid()
        self.c0 = None
        self.h0 = None

    def reset_lstm(self, input_dim):
        self.h0 = torch.zeros(1, self.hidden_dim)

    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        out = torch.relu(lstm_out[-1])
        out = self.hidden(out)
        out = self.sigmoid(out)
        return out


def train(config, train_data, tfidf, embeddings):
    def randomChoice(l):
        return l[random.randint(0, len(l) - 1)]

    device = torch.device(config['device'])
    model = LSTM(config).to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config['lr_decay'])

    positive_gradient_multiplier = config['positive_gradient_multiplier']

    depression_total = 0
    depression = 0
    correct = 0
    total = 0
    window_size = 200

    for iter in range(config['iter']):
        optimizer.zero_grad()
        current = randomChoice(train_data)
        stems = current[0]

        if len(stems) > window_size:
            start_pos = random.randint(0, len(stems) - window_size)
            stems = stems[start_pos: start_pos + window_size]

        embedding = embeddings(stems).view(-1, 1, 200)
        tfidfs = tfidf(stems)

        input = torch.cat((embedding, tfidfs.view(-1, 1, 1)), dim=2).to(device)
        output = model(input)

        label = current[1].float().view(1, 1).to(device)
        loss = criterion(output, label)

        if current[1] == 1:
            loss *= positive_gradient_multiplier
            depression_total += 1
            if output.item() > 0.5:
                depression += 1

        total += 1
        if abs(output.item() - label.item()) < 0.5:
            correct += 1

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if iter % 250 == 0:
            if depression_total != 0:
                print(
                    f"{iter} {iter / config['iter'] * 100:.2f}% Correct: {correct / total:.4f} Depression: {depression / depression_total:.4f}")
            else:
                print(f"{iter} {iter / config['iter'] * 100}% Correct: {correct / total} Depression: NaN")
            correct = 0
            total = 0
            depression = 0
            depression_total = 0

            if iter % 1000 == 0:
                scheduler.step()
    return model


def train_save(config, train_data, tfidf, name, data_dir, embeddings):
    model = train(config, train_data, tfidf, embeddings)
    data.save_model(model, config, data_dir + '/models', name)
    return model


def test(config, model, test_data, tfidf, embeddings, window_size, niter, verbose=False):

    if 'device' in config:
        device = torch.device(config['device'])
    else:
        device = torch.device('cpu')
    model = model.to(device)

    hits = {}
    hits_f = {}

    for i in range(len(test_data)):
        user, _, _ = test_data.get_full(i)
        hits[user] = 0
        hits_f[user] = 0

    def sig(x):
        return 1 / (1 + math.exp(5 - 10 * x))


    with torch.no_grad():
        for iter in range(niter):
            if verbose:
                print(f"Test round {iter + 1}")
            for i in range(len(test_data)):
                current = test_data.get_full(i)
                stems = current[1]
                if len(stems) == 0:
                    continue

                if len(stems) > window_size:
                    start_pos = random.randint(0, len(stems) - window_size)
                    stems = stems[start_pos: start_pos + window_size]

                embedding = embeddings(stems).view(-1, 1, 200)
                tfidfs = tfidf(stems)

                input = torch.cat((embedding, tfidfs.view(-1, 1, 1)), dim=2).to(device)
                output = model(input).item()

                hits_f[current[0]] += sig(output)

                if output > 0.5:
                    hits[current[0]] += 1

    return hits_f


def calcaulate_metrics(target, predictions, recall_bias=2.):
    recall = metrics.recall_score(target, predictions)
    precision = metrics.precision_score(target, predictions)
    f1 = (1 + recall_bias ** 2) * (precision * recall) / (recall_bias ** 2 * precision + recall)

    return recall, precision, f1

def hit_test(model_dir, name, test_data, tfidf, embeddings, tests, thresholds, window_size=None, verbose=False):
    model, config = data.load_model(model_dir, name, LSTM)
    window_size = config["window_size"] if window_size == None else window_size

    hits = test(config, model, test_data, tfidf, embeddings, window_size, tests, verbose=verbose)

    for threshold in thresholds:
        target = [x[1] for x in test_data]
        predictions = [0 if hits[test_data.get_full(x)[0]] < threshold else 1 for x in range(len(test_data))]

        recall, precision, f1 = calcaulate_metrics(target, predictions)

        print(f"Threshold \t{threshold:.4f}\n"
              f"\tPrecision \t{precision:.4f}\n"
              f"\tRecall \t\t{recall:.4f}\n"
              f"\tF1 \t\t\t{f1:.4f}")


if __name__ == "__main__":
    data_dir = str(Path(__file__).parent.parent / 'data')

    data_initialized = True
    glove_embeddings = False
    if not data_initialized:
        vocab = Vocab(data_dir + "/reddit-training-ready-to-share", -1, 5)
        data.pickle_data(data_dir + '/vocab.pic', vocab)

        if glove_embeddings:
            embedding_gen = WordVectorLoader(200)
            embedding_gen.load_glove(data_dir + '/glove.6B.200d.txt')
            embeddings = embedding_gen.generate_embedding_matrix(vocab, -1, 'random')
            torch.save(embeddings, data_dir + '/embeddings_glove.pic')
        else:
            word2vec_dim = 200
            w2v_file = data_dir + "/w2v_stopwords.model"
            word2vec = data.load_word2vec(data_dir)
            embedding_gen = WordVectorLoader(word2vec_dim)
            embedding_gen.set_word2vec(word2vec)
            embeddings = embedding_gen.generate_embedding_matrix(vocab, -1, 'random')
            torch.save(embeddings, data_dir + '/embeddings_w2v.pic')

        train_data = DatasetConcat(data_dir + "/reddit-training-ready-to-share", vocab)
        torch.save(train_data, data_dir + '/dataset_train.pic')
        test_data = DatasetConcat(data_dir + "/reddit-test-data-ready-to-share", vocab)
        torch.save(test_data, data_dir + '/dataset_test.pic')
    else:
        vocab = data.unpickle_data(data_dir + '/vocab.pic')
        if glove_embeddings:
            embeddings = torch.load(data_dir + '/embeddings.pic').float()
        else:
            embeddings = torch.load(data_dir + '/embeddings_w2v.pic').float()

        train_data = torch.load(data_dir + '/dataset_train.pic')
        test_data = torch.load(data_dir + '/dataset_test.pic')

    tfidf = TF_IDF(data.unpickle_data(data_dir + '/train.pic'), vocab)

    config = {}
    config['device'] = 'cuda:0'
    config['iter'] = 150000
    config['hidden'] = 128
    config['embedding_dim'] = 200
    config['lr'] = 2e-3
    config['lr_decay'] = 0.95
    config['window_size'] = 100
    config['weight_decay'] = 0.001
    config['lstm_layers'] = 1

    train_mode = False

    if train_mode:
        normal = len([0 for i in train_data if i[1] == 0])
        depression = len(train_data) - normal

#        config['positive_gradient_multiplier'] = math.log(normal / depression)
        config['positive_gradient_multiplier'] = 1

        train_save(config, train_data, tfidf, 'lstm_w2v_200', data_dir, embeddings)
    else:
        for name in ['lstm_w2v']:
            print("===================================")
            print("Testing: ", name)
            print("===================================")
            for ws in [100, 200]:
                print("Window size: ", ws)
                hit_test(data_dir + '/models', name, test_data, tfidf, embeddings, 20, [2, 2.5, 3, 3.5, 4, 4.5, 5], ws)
