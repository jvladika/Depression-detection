from pathlib import Path
import xml.etree.ElementTree as ET
import os

from string import punctuation
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import gensim
from gensim.models import Word2Vec
import torch
import pickle

DATA_DIR = str(Path(__file__).parent.parent / 'data')


def extract(directory):
    """
    Extracts data from the xml files in the directory
    :param directory: Directory with the data
    :return: Dictionary of users and their posts as lists
    """
    path = os.listdir(directory)

    users = {}
    for filename in path:
        tree = ET.parse(directory + "/" + filename)
        root = tree.getroot()
        id = root.find("ID").text
        users[id] = list()

        for w in root.findall("WRITING"):
            date = w.find("DATE").text
            text = w.find("TEXT").text
            users[id].append((date, text))

    return users


def train_word2vec(data_directory, w2v_size, exclude_stopwords=True):
    """
    Function for training the word2vec model with predefined parameters
    :param data_directory: Directory with data to train w2v on
    :param w2v_size: Size of the word2vec model
    :return: Trained word2vec model
    """
    positive_dir = data_directory + '/positive_examples_anonymous'
    negative_dir = data_directory + '/negative_examples_anonymous'

    positive_users = extract(positive_dir)
    negative_users = extract(negative_dir)

    def concatenate(users):
        result = {}
        for user, posts in users.items():
            result[user] = ''
            for post in posts:
                result[user] += post[1] + ' '

        return result

    positives = concatenate(positive_users)
    negatives = concatenate(negative_users)


    stemmer = SnowballStemmer("english")
    stop_words = list(map(lambda x: stemmer.stem(x), stopwords.words('english'))) + list(punctuation)

    from nltk.tokenize import word_tokenize

    for user, text in positives.items():
        positives[user] = word_tokenize(text)
        positives[user] = list(map(lambda x: stemmer.stem(x), positives[user]))
        if exclude_stopwords:
            positives[user] = list(filter(lambda x: x not in stop_words[35:], positives[user]))

    for user, text in negatives.items():
        negatives[user] = word_tokenize(text)
        negatives[user] = list(map(lambda x: stemmer.stem(x), negatives[user]))
        if exclude_stopwords:
            negatives[user] = list(filter(lambda x: x not in stop_words[35:], negatives[user]))

    sentences = []
    for k, v in positives.items():
        sentences.append(v)
    for k, v in negatives.items():
        sentences.append(v)

    return Word2Vec(sentences, size=w2v_size, window=10, min_count=5, workers=4, iter=15)


def load_word2vec(directory=DATA_DIR):
    """
    Shortcut function for loading the word2vec model from the directory.
    The word2vec model has to be saved with the filename w2v.model in the directory
    :param directory: Path to the directory to load the data from
    :return: Loaded word2vec model
    """
    return gensim.models.Word2Vec.load(directory + "/w2v.model")


def load_data(directory):
    """
        Function for loading data from a directory to a list of tuples
        :param directory: The root directory of the data (train, or test data directory)
        :return: Returns list of tuples (username, concatenated posts, label (0/1 normal or depressed))
    """

    positive_dir = directory + '/positive_examples_anonymous'
    negative_dir = directory + '/negative_examples_anonymous'

    positive_users = extract(positive_dir)
    negative_users = extract(negative_dir)

    data = []
    for user in positive_users:
        total = ""
        for text in positive_users[user]:
            total += text[1] + " "
        data.append((user, total, 1))

    for user in negative_users:
        total = ""
        for text in negative_users[user]:
            total += text[1] + " "
        data.append((user, total, 0))
    return data


def load_posts(directory, w2v_dir=DATA_DIR):
    """
        Function for loading data from a directory to a list of tuples
        :param directory: The root directory of the data (train, or test data directory)
        :return: Returns list of tuples (username, concatenated posts, label (0/1 normal or depressed))
    """

    w2v = load_word2vec(w2v_dir)
    stemmer = SnowballStemmer("english")

    positive_dir = directory + '/positive_examples_anonymous'
    negative_dir = directory + '/negative_examples_anonymous'

    positive_users = extract(positive_dir)
    negative_users = extract(negative_dir)

    data = []
    for user in positive_users:
        total = []
        for _, text in positive_users[user]:
            tokens = word_tokenize(text)
            stems = [stemmer.stem(x) for x in tokens]
            stems = list(filter(lambda x: x in w2v.wv, stems))
            if len(stems) > 0:
                total.append(stems)

        if len(total) > 0:
            data.append((user, total, 1))

    for user in negative_users:
        total = []
        for _, text in negative_users[user]:
            tokens = word_tokenize(text)
            stems = [stemmer.stem(x) for x in tokens]
            stems = list(filter(lambda x: x in w2v.wv, stems))
            if len(stems) > 0:
                total.append(stems)

        if len(total) > 0:
            data.append((user, total, 0))

    return data


def prepare_data(data, w2v_dir=DATA_DIR):
    """
    Function to prepare data for use. It takes a dictionary of users and their posts as an input,
    and returns a dictionary of users and their posts in a tokenized stemmed way.
    After stemming, all the words not active in the word2vec model are discarded.
    :param data: Dictionary of users and their posts concatenated to a string
    :param w2v_dir: Directory of the word2vec model used to prepare the data with. Default = DATA_DIR
    :return: Returns tokenized and stemmed data
    """

    w2v = load_word2vec(w2v_dir)
    stemmer = SnowballStemmer("english")

    data_c = []
    for current in data:
        tokens = word_tokenize(current[1])
        stems = [stemmer.stem(x) for x in tokens]
        stems = list(filter(lambda x: x in w2v.wv, stems))
        data_c.append((current[0], stems, current[2]))

    return data_c


def pickle_data(filepath, data):
    """
    Shortcut function for pickling data
    :param filepath: Path of file to save the pickled data to
    :param data: Data to save
    """
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def unpickle_data(filepath):
    """
    Shortcut function for unplickling data
    :param filepath: Filepath of file to unpickle
    :return: Unpickled data
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def init_data(data_dir = str(DATA_DIR), w2v_size=128, w2v=True, train=True, test=True):
    """
    :param data_dir: Data directory
    :param w2v_size: The size of the word2vec model
    :param w2v: Boolean flag, for training and saving word2vec
    :param train: Boolean flag, for processing training data
    :param test: Boolean flag, for processing test data
    """
    train_dir = data_dir + '/reddit-training-ready-to-share'
    test_dir = data_dir + '/reddit-test-data-ready-to-share'

    if w2v:
        print("Training word2vec")
        w2v = train_word2vec(train_dir, w2v_size)

        w2v_file = data_dir + "/w2v.model"
        w2v.save(w2v_file)
        print(f"word2vec model saved as '{w2v_file}'")

    if train:
        print("Loading train data")
        train_data = load_data(train_dir)
        print("Parsing train data")
        train_data = prepare_data(train_data)
        print("Saving train data")
        pickle_data(data_dir + '/train.pic', train_data)
    if test:
        print("Loading test data")
        test_data = load_data(test_dir)
        print("Parsing test data")
        test_data = prepare_data(test_data)
        print("Saving test data")
        pickle_data(data_dir + '/test.pic', test_data)


def load_config(config_directory, config_name):
    """
    Loads the config from the given directory
    :param config_directory: Directory of the config
    :param config_name: Name of the config (Without the extension as it is automatically added)
    :return: The configuration in the file
    """
    return unpickle_data(config_directory + '/' + config_name + '.config')


def save_model(model, config, model_directory, model_name):
    """
    Function for saving the model and config
    :param model: Model to save
    :param config: Config of the model to save
    :param model_directory: The directory where to save to
    :param model_name: The name under which to save the model and config files
    """
    path = model_directory + '/' + model_name
    #Save config
    pickle_data(path + '.config', config)
    torch.save(model.state_dict(), path + '.pt')


def load_model(model_directory, model_name, model_class):
    """
    Function for loading the model from file
    :param model_directory: The directory of the model
    :param model_name: The name of the model under which it has been saved
    :param model_class: The model class which inherits from the torch.nn.Module class
    :return: The loaded model and config
    """
    config = load_config(model_directory, model_name)
    model = model_class(config)
    model.load_state_dict(torch.load(model_directory + '/' + model_name +'.pt'))
    model.eval()
    return model, config

