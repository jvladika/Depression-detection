import math
import torch
import torch.nn as nn
import random
import data
from sklearn import metrics

class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.recurrent_size = config['recurrent']

        self.i2o = nn.Linear(config['w2v_size'] + config['recurrent'], config['hidden'])
        self.hidden = nn.Linear(config['hidden'], 1)

        self.i2h = nn.Linear(config['w2v_size'] + config['recurrent'], config['recurrent'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = torch.relu(output)
        output = self.hidden(output)
        output = self.sigmoid(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.recurrent_size)


class Connected(nn.Module):
    def __init__(self, config):
        super(Connected, self).__init__()
        self.input = nn.Linear(config['in'], config['hidden'])
        self.hidden = nn.Linear(config['hidden'], 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.input(input)
        output = torch.relu(output)
        output = self.hidden(output)
        output = self.sigmoid(output)
        return output

def train(config, train_data):
    w2v = data.load_word2vec()

    def randomChoice(l):
        return l[random.randint(0, len(l) - 1)]

    rnn = RNN(config)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config['lr_decay'])

    positive_gradient_multiplier = config['positive_gradient_multiplier']

    depression_total = 0
    depression = 0
    correct = 0
    total = 0

    label = torch.tensor(0, dtype=torch.float32).view(1, 1)

    for iter in range(config['iter']):
        current = randomChoice(train_data)
        stems = current[1]

        if len(stems) > config['window_size']:
            pos = random.randint(0, len(stems))
            stems = stems[pos: pos + config['window_size']]

        hidden = rnn.initHidden()
        optimizer.zero_grad()
        output = None

        for stem in stems:
            if stem in w2v.wv:
                input = torch.tensor(w2v.wv[stem]).view(1, -1)
                output, hidden = rnn(input, hidden)

        if output == None or math.isnan(output.item()):
            continue

        label[0][0] = current[2]
        loss = criterion(output, label)

        if current[2] == 1:
            loss *= positive_gradient_multiplier
            depression_total += 1
            if output > 0.5:
                depression += 1

        if math.fabs(current[2] - output.item()) < 0.5:
            correct += 1
        total += 1

        loss.backward()
        optimizer.step()

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

    return rnn


def train_save(config, train_data, name):
    model = train(config, train_data)
    data.save_model(model, config, data.DATA_DIR + '/baseline', name)
    return model


def test(model, test_data, window_size, niter, verbose=False):
    w2v = data.load_word2vec()

    hits = {}
    for user, _, _ in test_data:
        hits[user] = 0

    for iter in range(niter):
        if verbose:
            print(f"Test round {iter + 1}")
        for i, current in enumerate(test_data):
            stems = current[1]

            if len(stems) > window_size:
                pos = random.randint(0, len(stems))
                stems = stems[pos: pos + window_size]

            hidden = model.initHidden()
            output = None

            for stem in stems:
                output, hidden = model(torch.tensor(w2v.wv[stem]).view(1, -1), hidden)

            if output == None or math.isnan(output.item()):
                continue

            if output > 0.5:
                hits[current[0]] += 1

    return hits


def calcaulate_metrics(target, predictions, recall_bias=2.):
    recall = metrics.recall_score(target, predictions)
    precision = metrics.precision_score(target, predictions)
    f1 = (1 + recall_bias ** 2) * (precision * recall) / (recall_bias ** 2 * precision + recall)

    return recall, precision, f1


def hit_test(model_dir, name, test_data, tests, thresholds, window_size=None):
    model, config = data.load_model(model_dir, name, RNN)
    window_size = config["window_size"] if window_size == None else window_size
    hits = test(model, test_data, window_size, tests)

    for threshold in thresholds:
        target = [x[2] for x in test_data]
        predictions = [0 if hits[x[0]] < threshold else 1 for x in test_data]

        recall, precision, f1 = calcaulate_metrics(target, predictions)

        print(f"Threshold \t{threshold:.4f}\n"
              f"\tPrecision \t{precision:.4f}\n"
              f"\tRecall \t\t{recall:.4f}\n"
              f"\tF1 \t\t\t{f1:.4f}")


if __name__ == "__main__":

    config = {}
    config['iter'] = 50000
    config['hidden'] = 128
    config['recurrent'] = 128
    config['w2v_size'] = 128
    config['lr'] = 0.00005
    config['lr_decay'] = 0.975
    config['window_size'] = 200
    config['weight_decay'] = 0.001

    model = RNN(config)

    mode = 1

    if mode == 1:
        train_data = data.unpickle_data(data.DATA_DIR + '/train.pic')

        normal = len([0 for i in train_data if i[2] == 0])
        depression = len(train_data) - normal

        config['positive_gradient_multiplier'] = math.log(normal / depression) * 2
        train_save(config, train_data, "rnn_baseline_200_128")
    else:
        train_data = data.unpickle_data(data.DATA_DIR + '/train.pic')
        test_data = data.unpickle_data(data.DATA_DIR + '/test.pic')
        for ws in [50, 100, 200]:
            print(f'Window size: {ws}')
            hit_test(data.DATA_DIR + '/models/', "rnn_baseline_200_128", test_data, 20, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], ws)