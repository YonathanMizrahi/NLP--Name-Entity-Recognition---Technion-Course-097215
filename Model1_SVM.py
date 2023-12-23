import numpy as np
import re
from gensim.models import KeyedVectors
from sklearn.svm import SVC
import sklearn.metrics
from gensim import downloader


def get_data(path):
    """
    The function removes all non-alphanumeric characters from the words and converts all words to lowercase.
    The function returns a list of lists of words.
    :param path: path to the file
    :return: a list of lists of words.
    """
    with open(path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    sentences = [sen.strip().lower() for sen in sentences]
    sentences = [sen.split() for sen in sentences if sen]
    sentences = [[re.sub(r'\W+', '', w) for w in sen] for sen in sentences]
    return sentences

def arrange_labels(labels):
    """
    This function receive a list of labels and convert them to 0 or 1.
    :param labels: a list of labels
    :return: a converted list of labels
    """
    for i in range(len(labels)):
        if labels[i] == 'o':
            labels[i] = 0
        else:
            labels[i] = 1
    return labels

def split_text_and_labels(sentences, isDevCorpus = False):
    """
    This function split the sentences in words and labels
    :param sentences: words with labels
    :param isDevCorpus: a boolean which specifies if the input text comes from the dev.tagged file
    :return: words and labels
    """
    if isDevCorpus:
        # Remove the stop word from dev_tagged
        sentences = sentences[0:-1]
    text = [x[0] for x in sentences]
    labels = [x[1] for x in sentences]
    labels = arrange_labels(labels)
    return text,labels


def get_feature_representation(sentences, model):
    """
    This function takes a list of sentences and a glove model as input.
    It returns a list of vectors, where each vector is the feature representation of a word in the sentence.
    If a word is not in the corpus, its feature vector is set to 0.
    :param sentences: list of sentences
    :param model: a glove model
    :return: list of vectors
    """
    representation = []
    for i, word in enumerate(sentences):
        if word not in model.key_to_index:
            # If word isn't existing in the corpus, consider its feature vector to 0
            vec = [0] * 200
            representation.append(np.array(vec, dtype='f'))
        else:
            vec = model[word]
            representation.append(vec)
    representation = np.stack(representation)
    #tokenized_sen = representation
    return representation

def get_f1_score_model1():
    """
    This function is used to calculate the f1 score of the model 1.
    The model 1 is a SVM model with the glove twitter 200 model as the feature representation.
    The function will load the glove twitter 200 model, get the training and testing data,
    then it will get the feature representation of the training and testing data,
    After, it will initialize the SVM model, train the SVM model, predict the testing data,
    and calculate the f1 score.
    :return: f1 score
    """

    # Load the pre-trained glove twitter 200 model
    print('Importing glove twitter 200 model. ')
    GLOVE_PATH = 'glove-twitter-200'
    glove = downloader.load(GLOVE_PATH)
    model = glove
    print('Import finished.')
    model = KeyedVectors.load('gloveTwitter200.kv')

    # Get data
    train_sentences = get_data('data/train.tagged')
    dev_sentences = get_data('data/dev.tagged')

    #Split sentences in text and labels and arrange the labels to be 0 or 1
    train_text, train_labels = split_text_and_labels(train_sentences,isDevCorpus=False)
    dev_text, dev_labels = split_text_and_labels(dev_sentences, isDevCorpus=True)

    # Get the vector reprensentation for each word in the corpus
    trainFeature = get_feature_representation(train_text, model)
    devFeature = get_feature_representation(dev_text, model)

    # Initialize the SVM model and assign weights to each class
    print('Initializing the SVM model.')
    clf = SVC(kernel='rbf')
    # Train the SVM model
    print('Training the SVM model.')
    clf.fit(trainFeature, train_labels)
    # Use the model to predict the testing instances
    print('Predicting the labels.')
    y_pred = clf.predict(devFeature)
    # generate the classification report and calculate the f1 score
    print('Calculating the f1 score:')
    print('The classification report of our prediction is: ')
    print(sklearn.metrics.classification_report(dev_labels, y_pred))
    f1_score = sklearn.metrics.f1_score(dev_labels, y_pred)
    return f1_score

if __name__ == "__main__":
    f1_score = get_f1_score_model1()
    print('The f1 score on the file dev.tagged is: ' + str(f1_score))
