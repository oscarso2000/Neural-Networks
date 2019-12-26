import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
from torch.autograd import Variable
import time
from tqdm import tqdm
from data_loader import fetch_data
import spacy

unk = '<UNK>'


class RNN(nn.Module):
    def __init__(self, input_dim, h, output_size, n_layers, vocab_size, embedding_dim, device):  # Add relevant parameters
        super(RNN, self).__init__()
        # Fill in relevant parameters
        self.device = device
        self.h = h
        self.n_layers = n_layers
        self.i2h = nn.Linear(input_dim, h)
        # The rectified linear unit; one valid choice of activation function
        self.activation = nn.ReLU()
        self.fc = nn.Linear(h, output_size)
        self.rnn = nn.RNN(input_dim, h, 3, batch_first=True,
                          nonlinearity='relu')
        # Ensure parameters are initialized to small values, see PyTorch documentation for guidance
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()
        self.embed = nn.Embedding(vocab_size, embedding_dim)


    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        batch_size = inputs.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(inputs, hidden)

        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.h)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.h).to(self.device)
        return hidden


# You may find the functions make_vocab() and make_indices from ffnn.py useful; you are free to copy them directly (or call those functions from this file)
# Returns:
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index
        index2word[index] = word
    vocab.add(unk)
    return vocab, word2index, index2word


def word_embeddings(sequence):
    nlp = spacy.load("en_core_web_lg")
    token2index = {}
    index2token = {}
    sequence = nlp(sequence)
    print("Tokenizing sequences")
    for index, token in enumerate(sequence):
        token2index[token.vector] = index
        index2token[index] = token.vector
    return token2index, index2token

def other_embedding(data):
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab
    

# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    #vectorized_data = torch.zeros(len(data), 1, len(word2index))
    vectorized_data = []
    for document, y in data:
        vector = embed(document)
        vector = vector.unsqueeze(1)
    return vector
    """
    for document, y in data:
        #embedding = word_embeddings(str(document).lower())
        vector = torch.zeros(len(word2index),1,len(token2index))
        #vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            index2 = token2index.get(word.vector,token2index[unk.vector])
            #index2 = embedding.get(word, embedding[word.lower()])
            vector[index][0][index2] += 1
        vectorized_data.append((vector, y))
    return vectorized_data
    """


def main(hidden_dim, number_of_epochs):  # Add relevant parameters
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    print("Fetching data")
    # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    train_data, valid_data = fetch_data()
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)
    #token2index, index2token = word_embeddings(vocab) 
    print("Fetched and indexed data")
    print(token2index)
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    print("Vectorized data")
    # Create RNN

    model = RNN(input_dim=len(vocab), h=hidden_dim,
                output_size=5, n_layers=3, vocab_size = len(vocab), embedding_dim = 30*len(vocab), device=device)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print("Training for {} epochs.".format(number_of_epochs))
    for epoch in range(number_of_epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        # Good practice to shuffle order of training data
        random.shuffle(train_data)
        minibatch_size = 16
        N = len(train_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index *
                                                      minibatch_size + example_index]
                input_vector = input_vector.to(device)
                predicted_vector, hidden = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(
                    predicted_vector.view(1, -1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(
            epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))
        model.eval()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        # Good practice to shuffle order of training data #ERROR: RANDOM SHUFFLE VALID_DATA!!!!!
        random.shuffle(valid_data)
        minibatch_size = 16
        N = len(valid_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()  # they added this
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                input_vector = input_vector.to(device)
                predicted_vector, hidden = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(
                    predicted_vector.view(1, -1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))


    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this
"""
    while not stopping_condition:  # How will you decide to stop training and why
        optimizer.zero_grad()
        # You will need further code to operationalize training, ffnn.py may be helpful

        predicted_vector = model(input_vector)
        predicted_label = torch.argmax(predicted_vector)
        # You may find it beneficial to keep track of training accuracy or training loss;

        # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance

        # You will need to validate your model. All results for Part 3 should be reported on the validation set.
        # Consider ffnn.py; making changes to validation if you find them necessary
"""
