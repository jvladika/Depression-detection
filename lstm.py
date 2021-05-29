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

#        self.lstm = nn.LSTM(config['w2v_size'] + 1, config['hidden'], 1)
        self.lstm = nn.LSTM(config['w2v_size'] + 1, config['hidden'], config['lstm_layers'], bidirectional=False)

#        self.lstm = nn.LSTM(config['w2v_size'], config['hidden'])
        self.lstm.reset_parameters()
        self.hidden = nn.Linear(config['hidden'], 1)
        self.hidden.reset_parameters()
        self.sigmoid = nn.Sigmoid()
        self.c0 = None
        self.h0 = None

    def reset_lstm(self, input_dim):
        self.h0 = torch.zeros(1, self.hidden_dim)

    def forward(self, stems, w2v, tfidf):
#        input = torch.zeros((len(stems), 1, 128))
        input = torch.zeros((len(stems), 1, 129))
        for i, stem in enumerate(stems):
            input[i, 0, 0:128] = torch.from_numpy(w2v.wv[stem])
            input[i, 0, 128] =  tfidf[stem]
#            input[i, 0] = torch.from_numpy(w2v.wv[stem]) * tfidf[stem]
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
    config['lstm_layers'] = 1

    train_data = data.unpickle_data(data_dir + '/train.pic')
    tfidf = TF_IDF(data.unpickle_data(data_dir + '/train.pic'))

    train_mode = True

    if train_mode:
        normal = len([0 for i in train_data if i[2] == 0])
        depression = len(train_data) - normal

#        config['positive_gradient_multiplier'] = math.log(normal / depression)
        config['positive_gradient_multiplier'] = 1

        train_save(config, train_data, tfidf, 'lstm_single_no_positive_gradient_multiplier', data_dir)
    else:
        test_data = data.unpickle_data(data_dir + '/test.pic')
#        for name in ['lstm_single', 'lstm_single_third', 'double_lstm', 'double_lstm_2', 'lstm_test_cat']:
        for name in ['lstm_test_cat']:
            print("===================================")
            print("Testing: ", name)
            print("===================================")
            for ws in [200]:
                print("Window size: ", ws)
                hit_test(data_dir + '/models/lstm', name, test_data, tfidf, 20, [2, 2.5, 3, 3.5, 4, 4.5, 5], ws)

"""
Beta == 2
Tests: 20
window = 100
Tripple tested, variance = 0.5
lstm_single 100:
Threshold 	4.0000
	Precision 	0.3289
	Recall 		0.9074
	F1 			0.6712
Threshold 	4.5000
	Precision 	0.3588
	Recall 		0.8704
	F1 			0.6772
Threshold 	5.0000
	Precision 	0.3719
	Recall 		0.8333
	F1 			0.6677

Tests: 20
Quad tested, variance = 1.5
lstm_test_cat 100:
Threshold 	4.0000
	Precision 	0.3917
	Recall 		0.8704
	F1 			0.6994
Threshold 	4.5000
	Precision 	0.4112
	Recall 		0.8148
	F1 			0.6811
"""






"""
Beta == 2

===================================
Testing:  lstm_test_cat
===================================
Window size:  100  Tests:  1
Precision 	0.2537
Recall 		0.9444
F1 			0.6115
Window size:  200  Tests:  1
Precision 	0.3077
Recall 		0.8889
F1 			0.6452
Window size:  100  Tests:  1.5
Precision 	0.3311
Recall 		0.9074
F1 			0.6731
Window size:  200  Tests:  1.5
Precision 	0.3308
Recall 		0.7963
F1 			0.6214
Window size:  100  Tests:  2
Precision 	0.3607
Recall 		0.8148
F1 			0.6509
Window size:  200  Tests:  2
Precision 	0.4314
Recall 		0.8148
F1 			0.6918


===================================
Testing:  lstm_single
===================================
Window size:  100  Tests:  1
Precision 	0.2236
Recall 		0.9815
F1 			0.5850
Window size:  200  Tests:  1
Precision 	0.2772
Recall 		0.9444
F1 			0.6375
Window size:  100  Tests:  1.5
Precision 	0.2941
Recall 		0.9259
F1 			0.6477
Window size:  200  Tests:  1.5
Precision 	0.3287
Recall 		0.8704
F1 			0.6546
Window size:  100  Tests:  2
Precision 	0.3108
Recall 		0.8519
F1 			0.6319
Window size:  200  Tests:  2
Precision 	0.4151
Recall 		0.8148
F1 			0.6832
"""


"""
for name in ['lstm_test_cat']:
hit_test(data_dir + '/models', name, test_data, tfidf, 20, tests, ws)
===================================
Testing:  lstm_test_cat
===================================
Window size:  200  Tests:  3
Total samples 	 406	: 0.8768
Depression 		  54	: 0.7037
Normal 			 352	: 0.9034
Precision 	0.5278
Recall 		0.7037
F1 			0.6032
Window size:  200  Tests:  4
Total samples 	 406	: 0.8867
Depression 		  54	: 0.6481
Normal 			 352	: 0.9233
Precision 	0.5645
Recall 		0.6481
F1 			0.6034
"""

"""
===================================
Testing:  lstm_single
===================================
Window size:  200  Tests:  3
Total samples 	 406	: 0.7266
Depression 		  54	: 0.9074
Normal 			 352	: 0.6989
Precision 	0.3161
Recall 		0.9074
F1 			0.4689
Window size:  200  Tests:  4
Total samples 	 406	: 0.8079
Depression 		  54	: 0.8333
Normal 			 352	: 0.8040
Precision 	0.3947
Recall 		0.8333
F1 			0.5357
Window size:  200  Tests:  5
Total samples 	 406	: 0.8473
Depression 		  54	: 0.8148
Normal 			 352	: 0.8523
Precision 	0.4583
Recall 		0.8148
F1 			0.5867
Window size:  200  Tests:  6
Total samples 	 406	: 0.8744
Depression 		  54	: 0.7963
Normal 			 352	: 0.8864
Precision 	0.5181
Recall 		0.7963
F1 			0.6277
===================================
Testing:  lstm_single_third
===================================
Window size:  200  Tests:  3
Total samples 	 406	: 0.7956
Depression 		  54	: 0.8519
Normal 			 352	: 0.7869
Precision 	0.3802
Recall 		0.8519
F1 			0.5257
Window size:  200  Tests:  4
Total samples 	 406	: 0.8621
Depression 		  54	: 0.8333
Normal 			 352	: 0.8665
Precision 	0.4891
Recall 		0.8333
F1 			0.6164
Window size:  200  Tests:  5
Total samples 	 406	: 0.8768
Depression 		  54	: 0.7963
Normal 			 352	: 0.8892
Precision 	0.5244
Recall 		0.7963
F1 			0.6324
Window size:  200  Tests:  6
Total samples 	 406	: 0.8941
Depression 		  54	: 0.7222
Normal 			 352	: 0.9205
Precision 	0.5821
Recall 		0.7222
F1 			0.6446
===================================
Testing:  double_lstm
===================================
Window size:  200  Tests:  3
Total samples 	 406	: 0.7685
Depression 		  54	: 0.8519
Normal 			 352	: 0.7557
Precision 	0.3485
Recall 		0.8519
F1 			0.4946
Window size:  200  Tests:  4
Total samples 	 406	: 0.8054
Depression 		  54	: 0.7593
Normal 			 352	: 0.8125
Precision 	0.3832
Recall 		0.7593
F1 			0.5093
Window size:  200  Tests:  5
Total samples 	 406	: 0.8300
Depression 		  54	: 0.7037
Normal 			 352	: 0.8494
Precision 	0.4176
Recall 		0.7037
F1 			0.5241
Window size:  200  Tests:  6
Total samples 	 406	: 0.8473
Depression 		  54	: 0.6481
Normal 			 352	: 0.8778
Precision 	0.4487
Recall 		0.6481
F1 			0.5303
===================================
Testing:  double_lstm_2
===================================
Window size:  200  Tests:  3
Total samples 	 406	: 0.7808
Depression 		  54	: 0.8333
Normal 			 352	: 0.7727
Precision 	0.3600
Recall 		0.8333
F1 			0.5028
Window size:  200  Tests:  4
Total samples 	 406	: 0.8325
Depression 		  54	: 0.8148
Normal 			 352	: 0.8352
Precision 	0.4314
Recall 		0.8148
F1 			0.5641
Window size:  200  Tests:  5
Total samples 	 406	: 0.8350
Depression 		  54	: 0.6852
Normal 			 352	: 0.8580
Precision 	0.4253
Recall 		0.6852
F1 			0.5248
Window size:  200  Tests:  6
Total samples 	 406	: 0.8621
Depression 		  54	: 0.6667
Normal 			 352	: 0.8920
Precision 	0.4865
Recall 		0.6667
F1 			0.5625
===================================
Testing:  lstm_test_cat
===================================
Window size:  200  Tests:  3
Total samples 	 406	: 0.7734
Depression 		  54	: 0.7963
Normal 			 352	: 0.7699
Precision 	0.3468
Recall 		0.7963
F1 			0.4831
Window size:  200  Tests:  4
Total samples 	 406	: 0.8300
Depression 		  54	: 0.7963
Normal 			 352	: 0.8352
Precision 	0.4257
Recall 		0.7963
F1 			0.5548
Window size:  200  Tests:  5
Total samples 	 406	: 0.8448
Depression 		  54	: 0.7593
Normal 			 352	: 0.8580
Precision 	0.4505
Recall 		0.7593
F1 			0.5655
Window size:  200  Tests:  6
Total samples 	 406	: 0.8645
Depression 		  54	: 0.6852
Normal 			 352	: 0.8920
Precision 	0.4933
Recall 		0.6852
F1 			0.5736
"""