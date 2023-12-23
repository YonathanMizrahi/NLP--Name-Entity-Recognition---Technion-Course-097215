from Model3_bLSTM import reduceTag, read_data, prepare_sequence, get_spelling_feature
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
from sklearn.metrics import classification_report
#from gensim.models import KeyedVectors
from gensim import downloader


class NERDataset(Dataset):
    def __init__(self, data):
        # Retrieves longest sentence, for padding
        max_sentence_len = max([len(sentence) for sentence, tags in data])
        self.X = []
        self.X_original = []
        self.y = []
        self.X_spelling = []

        for sentence, tags in data:
            # Pad the sentences to the same length
            padded_sentence = sentence.copy()
            padded_tags = tags.copy()
            while len(padded_sentence) < max_sentence_len:
                padded_sentence.append('')
                padded_tags.append('')
            # Convert to indices
            transformed_sentence = prepare_sequence(padded_sentence, word_to_ix, use_unk=True)
            transformed_tags = prepare_sequence(padded_tags, tag_to_ix)
            # Get spelling indices
            spelling_sentence = get_spelling_feature(padded_sentence)
            # Add to dataset
            self.X.append(transformed_sentence)
            self.X_original.append(padded_sentence)
            self.y.append(transformed_tags)
            self.X_spelling.append(spelling_sentence)

        self.X = torch.from_numpy(np.array(self.X, dtype=np.int64)).to(device)
        self.y = torch.from_numpy(np.array(self.y, dtype=np.int64)).to(device)
        self.X_spelling = torch.from_numpy(np.array(self.X_spelling, dtype=np.int64)).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index], self.X_original[index], self.X_spelling[index]

class BLSTM2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, linear_dim, tags_size, lstm_dropout, elu_alpha,
                 embeddings, spelling_embedding_dim):
        super(BLSTM2, self).__init__()
        self.hidden_dim = hidden_dim

        self.embeddings_word = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=False,
                                                            padding_idx=word_to_ix[''])
        self.embeddings_spelling = nn.Embedding(num_embeddings=5, embedding_dim=spelling_embedding_dim, padding_idx=0)
        self.dropout_pre_lstm = nn.Dropout(lstm_dropout)
        self.lstm = nn.LSTM(embedding_dim + spelling_embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout_post_lstm = nn.Dropout(lstm_dropout)
        self.linear = nn.Linear(hidden_dim * 2, linear_dim)
        # self.elu = nn.ELU(alpha=elu_alpha)
        self.ReLu = nn.ReLU()
        self.linear2 = nn.Linear(linear_dim, tags_size)

    def forward(self, x_word, x_spelling):
        x1 = self.embeddings_word(x_word)
        x2 = self.embeddings_spelling(x_spelling)
        x = torch.cat((x1, x2), dim=2).to(device)
        x = self.dropout_pre_lstm(x)

        h0 = torch.zeros(2, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(2, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))

        out = self.dropout_post_lstm(out)
        out = self.linear(out)
        out = self.ReLu(out)
        # out = self.elu(out)
        out = self.linear2(out)

        return out

def predict2(model, data_loader):
    all_y = []
    all_y_pred = []
    model.eval()
    with torch.no_grad():
        for X, y, X_original, X_spelling in data_loader:
            X, y = X.to(device), y.to(device)

            y_pred_scores = model(X, X_spelling)
            y_pred = torch.argmax(y_pred_scores, dim=2)
            y_pred_flat = torch.flatten(y_pred).tolist()
            y_flat = torch.flatten(y).tolist()

            for i in range(len(y_pred_flat)):
                if y_flat[i] == tag_to_ix['']:
                    break
                all_y.append(y_flat[i])
                all_y_pred.append(y_pred_flat[i])
    print(classification_report(all_y, all_y_pred))

def create_file_test_tagged(model, sentences, fname):
    outputs = []
    model.eval()
    with torch.no_grad():
        for sentence in sentences:
            spelling_sentence = [get_spelling_feature(sentence)]
            spelling_sentence = torch.from_numpy(np.array(spelling_sentence, dtype=np.int64)).to(device)

            transformed_sentence = [prepare_sequence(sentence, word_to_ix, use_unk=True)]
            transformed_sentence = torch.from_numpy(np.array(transformed_sentence, dtype=np.int64)).to(device)

            y_pred_scores = model(transformed_sentence, spelling_sentence)
            y_pred = torch.argmax(y_pred_scores, dim=2)
            y_pred_flat = torch.flatten(y_pred).tolist()

            idx = 1
            output = []
            for i in range(len(y_pred_flat)):
                word = sentence[i]
                pred = ix_to_tag[y_pred_flat[i]]
                if word == '':
                    break
                output.append((word, pred))
                idx += 1
            outputs.append(output)

    with open(fname, 'w') as f:
        for i in range(len(outputs)):
            for j in range(len(outputs[i])):
                word, pred = outputs[i][j]
                f.write(f'{word} {pred}\n')
            if i != len(outputs) - 1:
                f.write('\n')
        f.write('\n')
        f.close()

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Let's load file train, dev and test and prepare them.")
    print('Note: Our model is trained on the file train.tagged,'
          'then the model will compute the f1 score on the file dev.tagged'
          'then the model will generate a file test.tagged')
    print('Device: ' + str(device))

    train_Path = 'data/train.tagged'
    test_Path = 'data/dev.tagged'
    eval_Path = 'data/test.untagged'

    # Read all datasets given
    train_data = read_data(train_Path)
    dev_data = read_data(test_Path)
    test_data = read_data(eval_Path, test_dataset=True)

    VOCAB_THRESHOLD = 0

    # Generate vocab
    words_freq = defaultdict(int)
    for sentence, tags in train_data:
        for word in sentence:
            words_freq[word] += 1

    vocab = {key for key, val in words_freq.items() if val >= VOCAB_THRESHOLD}

    # Generate word/tag to index mappings
    word_to_ix = {'': 0, '': 1}
    tag_to_ix = {'': 0}
    for sentence, tags in train_data:
        for word in sentence:
            if word not in vocab:
                word = ''
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    # Generate index to word/tag mappings
    ix_to_word = {v: k for k, v in word_to_ix.items()}
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}

    # Calculate the size of vocabulary & tags
    VOCAB_SIZE = len(word_to_ix)
    TAGS_SIZE = len(tag_to_ix)
    BATCH_SIZE = 1
    EMBEDDING_DIM = 200
    LSTM_HIDDEN_DIM = 256
    LSTM_DROPOUT = 0.25
    LINEAR_DIM = 164
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    ELU_ALPHA = 0.5
    SCHEDULER_STEP_SIZE = 5
    SCHEDULER_GAMMA = 0.5
    NUM_EPOCHS = 2
    SPELLING_EMBEDDING_DIM = 15
    # embeddings_dict = {}
    vocab = set(['', ''])

    """
    with open('glove.6B.300d.txt', 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
    """
    print('Importing glove twitter 200 model. ')
    GLOVE_PATH = 'glove-twitter-200'
    glove = downloader.load(GLOVE_PATH)
    embeddings_dict = glove
    print('Import finished.')
    #embeddings_dict = KeyedVectors.load('gloveTwitter200.kv')

    for sentence, tags in train_data:
        vocab.update(sentence)
    for sentence, tags in dev_data:
        vocab.update(sentence)
    for sentence in test_data:
        vocab.update(sentence)

    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {v: k for k, v in word_to_ix.items()}

    embedding_matrix = np.zeros((len(vocab), EMBEDDING_DIM))

    for word in vocab:
        index = word_to_ix[word]
        if word in embeddings_dict:
            vector = embeddings_dict[word]
        elif word.lower() in embeddings_dict:
            vector = embeddings_dict[word.lower()]
        else:
            vector = np.random.rand(EMBEDDING_DIM)
        embedding_matrix[index] = vector

    VOCAB_SIZE = len(word_to_ix)

    """# Data"""
    train_dataset = NERDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    dev_dataset = NERDataset(dev_data)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False)

    model = BLSTM2(VOCAB_SIZE, EMBEDDING_DIM, LSTM_HIDDEN_DIM, LINEAR_DIM, TAGS_SIZE, LSTM_DROPOUT, ELU_ALPHA,
                   embedding_matrix, SPELLING_EMBEDDING_DIM).to(device)

    ratio = float(2000 / 14483)

    weights = [0, ratio, 1 - ratio]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=MOMENTUM)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    # print('Training a new model...')
    total_loss = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        for i, (X, y, X_original, X_spelling) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            y_pred_scores = model(X, X_spelling)
            y_pred = torch.flatten(y_pred_scores, start_dim=0, end_dim=1)
            y = torch.flatten(y)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
        total_loss = []
        scheduler.step()
        print(epoch)
        predict2(model, dev_loader)
    torch.save(model, 'CustomModel_CPU.pt')
    # print('model saved')
    print("Now let's test the saved model2: CustomModel_CPU")
    model2 = torch.load('CustomModel_CPU.pt')
    print('Classification Report (including f1 score) on file dev.tagged')
    predict2(model2, dev_loader)
    print("Let's now create our comp tagged file")
    create_file_test_tagged(model2, test_data, 'comp_209948728_931188684.tagged')



