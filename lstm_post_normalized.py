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

    def forward(self, batch, w2v, tfidf):
        max_len = len(max(batch, key=lambda x:len(x)))
        input = torch.zeros((max_len, len(batch), 128))
        for i, posts in enumerate(batch):
            for j, post in enumerate(posts):
                for word in post:
                    if word in w2v.wv:
                        input[j, i] += torch.from_numpy(w2v.wv[word]) * tfidf[word]
                input[j, i] /= torch.sum(input[j, i])

#            input[i, 0] = torch.from_numpy(w2v.wv[stem])
        lstm_out, _ = self.lstm(input)
        out = torch.relu(lstm_out[-1, :, :])
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

    depression_total = 0
    depression = 0
    correct = 0
    total = 0

    max_posts = 25
    batch_size = 5
    labels = torch.zeros(batch_size, 1)

    optimizer.zero_grad()
    for iter in range(config['iter']):
        posts_batch = []

        for i in range(batch_size):
            current = randomChoice(train_data)
            user = current[0]
            posts = current[1]

            if len(posts) > max_posts:
                start_pos = random.randint(0, len(posts) - max_posts)
                posts = posts[start_pos : start_pos + max_posts]
            posts_batch.append(posts)
            labels[i] = current[2]

        optimizer.zero_grad()

        output = model(posts_batch, w2v, tfidf)
        loss = criterion(output, labels)

        loss.backward()
        optimizer.step()

        diff = torch.abs(output - labels)
        total += batch_size
        correct += len(diff[diff < 0.5])
        depression_mask = labels > 0.5
        depressive = output[depression_mask]
        depression_total += len(depressive)
        depression += len(depressive[depressive >= 0.5])



        if iter % 50 == 0:
            if depression_total != 0:
                print(
                    f"{iter} {iter / config['iter'] * 100:.2f}% Correct: {correct / total:.4f} Depression: {depression / depression_total:.4f}")
            else:
                print(f"{iter} {iter / config['iter'] * 100}% Correct: {correct / total} Depression: NaN")
            correct = 0
            total = 0
            depression = 0
            depression_total = 0

            if iter % 100 == 0:
                scheduler.step()
    return model


def train_save(config, train_data, tfidf, name, data_dir):
    model = train(config, train_data, tfidf)
    data.save_model(model, config, data_dir + '/models', name)
    return model


def test(model, test_data, tfidf, window_size, niter, verbose=False):
    w2v = data.load_word2vec()

    hits = {}
    hits_f = {}

    for user, _, _ in test_data:
        hits[user] = 0
        hits_f[user] = 0

    def sig(x):
        return 1/(1 + math.exp(5 - 10 * x))

    for iter in range(niter):
        if verbose:
            print(f"Test round {iter + 1}")
        for i, current in enumerate(test_data):
            post = current[1]
            if len(post) == 0:
                continue
            if len(post) > window_size:
                pos = random.randint(0, len(post)-window_size)
                post = post[pos: pos + window_size]

            output = model([post], w2v, tfidf)
            hits_f[current[0]] += sig(output.item())
            if output.item() > 0.5:
                hits[current[0]] += 1

    return hits_f


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
    data_dir = str(Path(__file__).parent.parent / 'data')

    data_initialized = True
    if not data_initialized:
        posts_train = data.load_posts(data_dir + "/reddit-training-ready-to-share")
        posts_test = data.load_posts(data_dir + "/reddit-test-data-ready-to-share")
        data.pickle_data(data_dir + '/posts_train.pic', posts_train)
        data.pickle_data(data_dir + '/posts_test.pic', posts_test)

    config = {}
    config['iter'] = 10000
    config['hidden'] = 64
    config['w2v_size'] = 128
    config['lr'] = 5e-2
    config['lr_decay'] = 0.95
    config['window_size'] = 100
    config['weight_decay'] = 0.001

    train_data = data.unpickle_data(data_dir + '/posts_train.pic')
    tfidf = TF_IDF(data.unpickle_data(data_dir + '/train.pic'))

    train_mode = True

    if train_mode:
        normal = len([0 for i in train_data if i[2] == 0])
        depression = len(train_data) - normal

        config['positive_gradient_multiplier'] = math.log(normal / depression)
#        config['positive_gradient_multiplier'] = 1

        train_save(config, train_data, tfidf, 'lstm_batched', data_dir)
    else:
        test_data = data.unpickle_data(data_dir + '/posts_test.pic')

        for name in ['lstm_batched_2']:
            print("===================================")
            print("Testing: ", name)
            print("===================================")
            hit_test(data_dir + '/models/lstm', name, test_data, tfidf, 20, [2, 2.5, 3.0, 3.5, 4.0, 4.5, 5], 25)

#            for tests in [1, 1.5, 2, 2.5, 3]:
#                for ws in [200]:
#                    print("Window size: ", ws, " Tests: ", tests)
#                    hit_test(data_dir + '/models', name, test_data, tfidf, 1, tests, 25)
