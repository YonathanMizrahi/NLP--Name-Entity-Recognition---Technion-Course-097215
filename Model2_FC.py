import numpy as np
import re
import sklearn.metrics
from gensim.models import KeyedVectors
import torch
from torch import nn
from torch.utils.data import Dataset
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam

from gensim import downloader


class SentimentDataSet(Dataset):
    def __init__(self, data):
        self.sentences = data['reviewText'].tolist()
        self.labels = data['label'].tolist()
        self.tags_to_idx = {tag: idx for idx, tag in enumerate(sorted(list(set(self.labels))))}
        self.idx_to_tag = {idx: tag for tag, idx in self.tags_to_idx.items()}

        #model = KeyedVectors.load('gloveTwitter200.kv')



        representation, labels = [], []
        for sen, cur_labels in zip(self.sentences, self.labels):
            cur_rep = []
            for word in sen.split():
                word = re.sub(r'\W+', '', word.lower())
                isExistASimilarWord = False
                if word not in model.key_to_index:
                    vec = [0] * 200
                    """
                    for i in range(1,len(word)):
                        try:
                            sims = model.most_similar(word[:-i], topn=5)
                            isExistASimilarWord = True
                            if isExistASimilarWord:
                                break
                        except:
                            continue
                    if isExistASimilarWord:
                        vec = [0] * 200
                        vec += model[sims[0][0]]
                        vec += model[sims[1][0]]
                        vec += model[sims[2][0]]
                        vec += model[sims[3][0]]
                        vec += model[sims[4][0]]
                        vec /= 5
                    else:
                        vec = [0] * 200
                    """
                else:
                    vec = model[word]
                cur_rep.append(vec)
            if len(cur_rep) == 0:
                # print(f'Sentence {sen} cannot be represented!')
                continue
            cur_rep = np.stack(cur_rep[0])
            representation.append(cur_rep)
            labels.append(cur_labels)
        self.labels = labels
        representation = np.stack(representation)
        self.tokenized_sen = representation
        self.vector_dim = representation.shape[-1]

    def __getitem__(self, item):
        cur_sen = self.tokenized_sen[item]
        cur_sen = torch.FloatTensor(cur_sen).squeeze()
        label = self.labels[item]
        label = self.tags_to_idx[label]
        data = {"input_ids": cur_sen, "labels": label}
        return data

    def __len__(self):
        return len(self.labels)


class SentimentNN(nn.Module):
    def __init__(self, vec_dim, num_classes, hidden_dim=100):
        super(SentimentNN, self).__init__()
        self.first_layer = nn.Linear(vec_dim, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.Tanh()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        x = self.first_layer(input_ids)
        x = self.activation(x)
        x = self.second_layer(x)
        if labels is None:
            return x, None
        loss = self.loss(x, labels)
        return x, loss

def get_word_and_labels(path):
    """
    This function takes a path to a file and returns a dataframe with two columns:
        reviewText: the words in the file
        label: 0 or 1
    :param path: path to the file
    :return: a dataframe with two columns
    """
    tags_list = []
    words_list = []
    with open(path,errors="ignore") as f:
        for i,l in enumerate(f.read().splitlines()):
            if l != '\t' and l != '':
                line = l.split("\t")
                if len(line) > 1:
                    tag = line[1]
                    word = line[0]
                if tag == 'O':
                    tags_list.append(0)
                else:
                    tags_list.append(1)
                words_list.append(word)
    df = pd.DataFrame({'reviewText': words_list,'label': tags_list})
    return df



def train(model, data_sets, optimizer, num_epochs: int, batch_size=16):
    """
    Train the model and print the loss and f1 score for each epoch on the train and test set
    :param model: The model to be trained.
    :param data_sets: A dictionary containing the training and test data sets.
    :param optimizer: The optimizer to be used for training.
    :param num_epochs: The number of epochs to train for.
    :param batch_size: The batch size to use for training.
    :return: None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loaders = {"train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
                    "test": DataLoader(data_sets["test"], batch_size=batch_size, shuffle=False)}
    model.to(device)

    best_acc = 0.0
    best_f1_score = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            labels, preds = [], []

            for batch in data_loaders[phase]:
                batch_size = 0
                for k, v in batch.items():
                    batch[k] = v.to(device)
                    batch_size = v.shape[0]

                optimizer.zero_grad()
                if phase == 'train':
                    outputs, loss = model(**batch)
                    loss.backward()
                    optimizer.step()
                else:
                    with torch.no_grad():
                        outputs, loss = model(**batch)
                pred = outputs.argmax(dim=-1).clone().detach().cpu()
                labels += batch['labels'].cpu().view(-1).tolist()
                preds += pred.view(-1).tolist()
                running_loss += loss.item() * batch_size
            epoch_loss = running_loss / len(data_sets[phase])
            epoch_f1_score = sklearn.metrics.f1_score(labels, preds)
            epoch_f1_score = round(epoch_f1_score, 5)

            if phase.title() == "test":
                print(f'{phase.title()} Loss: {epoch_loss:.4e} F1 Score: {epoch_f1_score}')
            else:
                print(f'{phase.title()} Loss: {epoch_loss:.4e} F1 Score: {epoch_f1_score}')
            if phase == 'test' and epoch_f1_score > best_f1_score:
                best_f1_score = epoch_f1_score
                with open('model.pkl', 'wb') as f:
                    torch.save(model, f)
        print()
    print('----------')
    print(f'Best Validation F1 Score on the test set during the train: {best_f1_score:4f}')

def test(model, data_sets):
    """
    Test the model and print the f1 score on all the test set.
    :param model: The pre-trained model .
    :return: None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = len(data_sets["test"])
    data_loaders = {"train": DataLoader(data_sets["train"], batch_size=batch_size, shuffle=True),
                    "test": DataLoader(data_sets["test"], batch_size=batch_size, shuffle=False)}
    labels, preds = [], []
    model.to(device)
    model.eval()
    for batch in data_loaders['test']:
        with torch.no_grad():
            outputs, loss = model(**batch)
        pred = outputs.argmax(dim=-1).clone().detach().cpu()
        labels += batch['labels'].cpu().view(-1).tolist()
        preds += pred.view(-1).tolist()
    f1_score = sklearn.metrics.f1_score(labels, preds)
    f1_score = round(f1_score, 5)
    print(f'The F1 Score on all the dev.tagged file is: {f1_score:4f}')

def get_f1_score_model2():
    """
        This function returns the F1 score of the model.
        The model is a simple neural network with one hidden layer.
        The model is trained on the training set and tested on the dev set.
        The model is trained for 5 epochs.
        The model is trained using the Adam optimizer.
        The function will print the loss and f1 score for each epoch on the train and test set.
        The function will also print the f1 score on all the test set.
    """
    train_tagged = get_word_and_labels('data/train.tagged')
    dev_tagged = get_word_and_labels('data/dev.tagged')

    train_ds = SentimentDataSet(train_tagged)
    print('Created train')
    test_ds = SentimentDataSet(dev_tagged)
    datasets = {"train": train_ds, "test": test_ds}
    nn_model = SentimentNN(num_classes=2, vec_dim=train_ds.vector_dim)
    optimizer = Adam(params=nn_model.parameters())
    train(model=nn_model, data_sets=datasets, optimizer=optimizer, num_epochs=5)
    test(model=nn_model, data_sets=datasets)

if __name__ == "__main__":
    print('Importing glove twitter 200 model. ')
    GLOVE_PATH = 'glove-twitter-200'
    glove = downloader.load(GLOVE_PATH)
    model = glove
    print('Import finished.')
    get_f1_score_model2()