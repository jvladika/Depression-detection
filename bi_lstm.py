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
import copy

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

        self.lstm = nn.LSTM(config['w2v_size'] + 1, config['hidden'], 2, bidirectional=True)
        self.hidden = nn.Linear(config['hidden'] * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, stems, w2v, tfidf):
        input = torch.zeros((len(stems), 1, 129))
        for i, stem in enumerate(stems):
            input[i, 0, 0:128] = torch.from_numpy(w2v.wv[stem])
            input[i, 0, 128] =  tfidf[stem]
        lstm_out, _ = self.lstm(input)
        out = torch.relu(lstm_out[-1])
        out = self.hidden(out)
        out = self.sigmoid(out)
        return out


def train(config, train_data, tfidf):

    def randomChoice(l):
        return l[random.randint(0, len(l) - 1)]


    train_data, valid_data = split_dataset(train_data, 0.8)


    w2v = data.load_word2vec()
    model = LSTM(config)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config['lr_decay'])

    positive_gradient_multiplier = config['positive_gradient_multiplier']

    label = torch.zeros((1, 1))
    window_size = 200

    optimizer.zero_grad()

    depression_total = 0
    depression = 0
    correct = 0
    total = 0

    best_f1 = None
    best_f1_score = 0
    best_recall = None
    best_recall_score = 0

    training = True

    for iter in range(config['iter']):
        if not training:
            break

        optimizer.zero_grad()

        current = randomChoice(train_data)
        stems = current[1]
        label[0][0] = current[2]

        if len(stems) > window_size:
            start_pos = random.randint(0, len(stems) - window_size)
            stems = stems[start_pos : start_pos + window_size]

        output = model(stems, w2v, tfidf)
        loss = criterion(output, label)

        if current[2] == 1:
            loss *= positive_gradient_multiplier

        loss.backward()
        optimizer.step()

        if current[2] == 1:
            loss *= positive_gradient_multiplier
            depression_total += 1
            if output.item() > 0.5:
                depression += 1

        total += 1
        if abs(output.item() - label.item()) < 0.5:
            correct += 1

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

                if iter % 5000 ==  0 and iter != 0:
                    recall, f1 = hit_test_data(model, valid_data, tfidf, 200, 20, 5)
                    print("========================================")
                    print(f"{iter}. F1: {f1:.4f} Recall: {recall:.4f}")
                    print("========================================")

                    if f1 > best_f1_score:
                        best_f1 = copy.deepcopy(model)
                    if recall > best_recall_score:
                        best_recall = copy.deepcopy(model)

    return model, best_recall, best_f1


def train_save(config, train_data, tfidf, name, data_dir):
    model, best_recall, best_f1 = train(config, train_data, tfidf)
    data.save_model(model, config, data_dir + '/models', name)
    data.save_model(best_recall, config, data_dir + '/models', name + '_recall')
    data.save_model(best_f1, config, data_dir + '/models', name + '_f1')
    return model


def hit_test_data(model, test_data, tfidf, window_size, tests, treshold):
    hits = test(model, test_data, tfidf, window_size, tests)

    target = [x[2] for x in test_data]
    predictions = [0 if hits[x[0]] < treshold else 1 for x in test_data]

    recall = metrics.recall_score(target, predictions)
    f1 = metrics.f1_score(target, predictions)
    return recall, f1


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
            stems = current[1]
            if len(stems) == 0:
                continue
            if len(stems) > window_size:
                pos = random.randint(0, len(stems)-window_size)
                stems = stems[pos: pos + window_size]

            output = model(stems, w2v, tfidf)
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


def split_dataset(dataset, split: float):
    positives = [instance for instance in dataset if instance[2] > 0.5]
    negatives = [instance for instance in dataset if instance[2] < 0.5]

    positives_split = int(math.floor(len(positives) * split))
    negatives_split = int(math.floor(len(negatives) * split))

    set_1 = positives[0:positives_split] + negatives[0:negatives_split]
    set_2 = positives[positives_split:] + negatives[negatives_split:]

    return set_1, set_2



if __name__ == "__main__":
    data_dir = str(Path(__file__).parent.parent / 'data')

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
    config['lr'] = 1e-2
    config['lr_decay'] = 0.95
    config['window_size'] = 100
    config['weight_decay'] = 0.001

    train_data = data.unpickle_data(data_dir + '/train.pic')
    tfidf = TF_IDF(data.unpickle_data(data_dir + '/train.pic'))

    train_mode = True

    if train_mode:
        normal = len([0 for i in train_data if i[2] == 0])
        depression = len(train_data) - normal

        config['positive_gradient_multiplier'] = math.log(normal / depression)
#        config['positive_gradient_multiplier'] = 1

        train_save(config, train_data, tfidf, 'bilstm', data_dir)
    else:
        test_data = data.unpickle_data(data_dir + '/test.pic')
        for name in ['bilstm']:
            print("===================================")
            print("Testing: ", name)
            print("===================================")
            for ws in [100]:
                print("Window size: ", ws)
                hit_test(data_dir + '/models/bi_lstm', name, test_data, tfidf, 10, [1, 1.5, 2, 2.5, 3], ws)

"""
Bilstm:
Double tested
F1 beta = 2
Best: 
    Tests 10, treshold 1.5
    Precision 	0.4659
	Recall 		0.7593
	F1 			0.6743
"""



"""
===================================
Testing:  bilstm
===================================
Window size:  200  Tests:  3
Precision 	0.4937
Recall 		0.7222
F1 			0.5865
Window size:  200  Tests:  4
Precision 	0.5645
Recall 		0.6481
F1 			0.6034
Window size:  200  Tests:  5
Precision 	0.6327
Recall 		0.5741
F1 			0.6019
Window size:  200  Tests:  6
Precision 	0.6279
Recall 		0.5000
F1 			0.5567
===================================
Testing:  bilstm_f1
===================================
Window size:  200  Tests:  3
Precision 	0.4930
Recall 		0.6481
F1 			0.5600
Window size:  200  Tests:  4
Precision 	0.6444
Recall 		0.5370
F1 			0.5859
Window size:  200  Tests:  5
Precision 	0.6410
Recall 		0.4630
F1 			0.5376
Window size:  200  Tests:  6
Precision 	0.6774
Recall 		0.3889
F1 			0.4941
===================================
Testing:  bilstm_recall
===================================
Window size:  200  Tests:  3
Precision 	0.5000
Recall 		0.6481
F1 			0.5645
Window size:  200  Tests:  4
Precision 	0.6226
Recall 		0.6111
F1 			0.6168
Window size:  200  Tests:  5
Precision 	0.6923
Recall 		0.5000
F1 			0.5806
"""