import data
from pathlib import Path
import itertools
import collections
import torch.nn as nn
import torch
import math
import numpy as np
import random
from sklearn import metrics

class TF_IDF:
    def __init__(self, data, w2v=data.load_word2vec()):
        self.data = data
        depressed_data = [x for x in self.data if x[2] == 1]

        stems = list(itertools.chain.from_iterable([x[1] for x in self.data]))
        stems = [x for x in stems if x in w2v.wv]
        word_frequencies = collections.Counter(stems)

        d_stems = list(itertools.chain.from_iterable([x[1] for x in depressed_data]))
        d_stems = [x for x in d_stems if x in w2v.wv]
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
            self.idf_scores[word] = math.log(documents / word_frequencies[word]) * math.tanh(math.pow((word_frequencies[word]) * 0.025, 3))

        tfidf_scores = {}

        self.min_tf_score = self.tf_scores[min(self.tf_scores, key=lambda x: self.tf_scores[x])]

        for word in self.idf_scores:
            if word in self.tf_scores:
                tfidf_scores[word] = self.tf_scores[word] * self.idf_scores[word]
            else:
                tfidf_scores[word] = self.min_tf_score * self.idf_scores[word]

        self.scores = tfidf_scores

    def get(self, x):
        if x in self.scores:
            return self.scores[x]
        if x in self.idf_scores:
            self.scores[x] = self.min_tf_score * self.idf_scores[x]
            return self.scores[x]

    def __getitem__(self, key):
        return self.get(key)


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.hidden_dim = config['hidden']

        self.lstm = nn.LSTM(config['w2v_size'], config['hidden'])
        self.lstm.reset_parameters()
        self.hidden = nn.Linear(config['hidden'], 1)
        self.hidden.reset_parameters()
        self.sigmoid = nn.Sigmoid()
        self.c0 = None
        self.h0 = None

    def reset_lstm(self, input_dim):
        self.h0 = torch.zeros(1, self.hidden_dim)

    def forward(self, stems, w2v, tfidf):
        input = torch.zeros((len(stems), 1, 128))
        for i, stem in enumerate(stems):
            input[i, 0] = torch.from_numpy(w2v.wv[stem]) * tfidf[stem]
        lstm_out, _ = self.lstm(input)
        out = torch.relu(lstm_out[-1])
        out = self.hidden(out)
        out = self.sigmoid(out)
        return out


def train(config, train_data, tfidf):

    def randomChoice(l):
        return l[random.randint(0, len(l) - 1)]

    w2v = data.load_word2vec()
    model = LSTM(config)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config['lr_decay'])

    positive_gradient_multiplier = config['positive_gradient_multiplier']
    w2v_size = config['w2v_size']

    depression_total = 0
    depression = 0
    correct = 0
    total = 0

    label = torch.zeros((1, 1))
    window_size = 200

    optimizer.zero_grad()
    for iter in range(config['iter']):
        current = randomChoice(train_data)
        user = current[0]
        stems = current[1]
        label[0][0] = current[2]

        if len(stems) > window_size:
            start_pos = random.randint(0, len(stems) - window_size)
            stems = stems[start_pos : start_pos + window_size]
#        post = randomChoice(posts)


        output = model(stems, w2v, tfidf)
        loss = criterion(output, label)

        if current[2] == 1:
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


def train_save(config, train_data, tfidf, name, data_dir):
    model = train(config, train_data, tfidf)
    data.save_model(model, config, data_dir + '/models', name)
    return model


def test(model, test_data, tfidf, window_size, niter, verbose=False):
    w2v = data.load_word2vec()

    hits = {}

    for user, _, _ in test_data:
        hits[user] = 0

    for iter in range(niter):
        if verbose:
            print(f"Test round {iter + 1}")
        for i, current in enumerate(test_data):
            stems = current[1]
            if len(stems) == 0:
                continue
            if len(stems) > window_size:
                pos = random.randint(0, len(stems)-window_size)
                stems = stems[pos: pos + window_size]

            output = model(stems, w2v, tfidf)
            if output.item() > 0.5:
                hits[current[0]] += 1

    return hits


def calcaulate_metrics(target, predictions, recall_bias = 2.):
    recall = metrics.recall_score(target, predictions)
    precision = metrics.precision_score(target, predictions)
    f1 = (1 + recall_bias**2) * (precision * recall) / (recall_bias**2 * precision + recall)

    return recall, precision, f1


def hit_test(model_dir, name, test_data, tfidf, tests, thresholds, window_size = None, verbose=False):
    model, config = data.load_model(model_dir, name, LSTM)
    window_size = config["window_size"] if window_size == None else window_size

    hits = test(model, test_data, tfidf, window_size, tests, verbose=verbose)

    for threshold in thresholds:

        target = [x[2] for x in test_data]
        predictions = [0 if hits[x[0]] < threshold else 1 for x in test_data]

        recall, precision, f1 = calcaulate_metrics(target, predictions)

        print(f"Threshold \t{threshold:.4f}\n"
              f"\tPrecision \t{precision:.4f}\n"
              f"\tRecall \t\t{recall:.4f}\n"
              f"\tF1 \t\t\t{f1:.4f}")


if __name__ == "__main__":
    data_dir = str(Path(__file__).parent.parent.parent / 'data')

    data_initialized = True
    if not data_initialized:
        posts_train = data.load_posts(data_dir + "/reddit-training-ready-to-share")
        posts_test = data.load_posts(data_dir + "/reddit-test-data-ready-to-share")
        data.pickle_data(data_dir + '/posts_train.pic', posts_train)
        data.pickle_data(data_dir + '/posts_test.pic', posts_test)

    config = {}
    config['iter'] = 100000
    config['hidden'] = 64
    config['w2v_size'] = 128
    config['lr'] = 5e-3
    config['lr_decay'] = 0.95
    config['window_size'] = 100
    config['weight_decay'] = 0.001

    train_data = data.unpickle_data(data_dir + '/train.pic')
    tfidf = TF_IDF(data.unpickle_data(data_dir + '/train.pic'))

    train_mode = False

    if train_mode:
        normal = len([0 for i in train_data if i[2] == 0])
        depression = len(train_data) - normal

        config['positive_gradient_multiplier'] = math.log(normal / depression)
#        config['positive_gradient_multiplier'] = 1

        train_save(config, train_data, tfidf, 'lstm_tfidf_mul', data_dir)
    else:
        test_data = data.unpickle_data(data_dir + '/test.pic')
        for name in ['lstm_tfidf_mul']:
            print("===================================")
            print("Testing: ", name)
            print("===================================")
            #for tests in [3, 3.5, 4, 4.5, 5, 5.5, 6]:
            for ws in [100, 200]:
                print("Window size: ", ws)
                hit_test(data_dir + '/models/lstm', name, test_data, tfidf, 20, [3, 3.5, 4, 4.5, 5], ws)

"""
Losi rezultati
60 f1 beta 2
"""