import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
from pathlib import Path 
import time
from tqdm import tqdm
from data_loader import fetch_data

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
	def __init__(self, input_dim, h):
			super(FFNN, self).__init__()
			self.h = h
			self.W1 = nn.Linear(input_dim, h)
			self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
			self.W2 = nn.Linear(h, 5) # ERROR 3: nxd matrix, d dimension should be 5 instead of h
			# The below two lines are not a source for an error
			self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
			self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

	def compute_Loss(self, predicted_vector, gold_label):
		return self.loss(predicted_vector, gold_label)

	def forward(self, input_vector):
		# The z_i are just there to record intermediary computations for your clarity
		z1 = self.W1(input_vector)
		z2 = self.W2(z1)
		predicted_vector = self.softmax(self.activation(z2))
		return predicted_vector


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
	vocab.add(unk) #ERROR 1: change 'unk' to unk variable
	return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
	vectorized_data = []
	for document, y in data:
		vector = torch.zeros(len(word2index)) 
		for word in document:
			index = word2index.get(word, word2index[unk])
			vector[index] += 1
		vectorized_data.append((vector, y))
	return vectorized_data


def main(hidden_dim, number_of_epochs):
	print("Fetching data")
	train_data, valid_data = fetch_data() # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
	vocab = make_vocab(train_data)
	vocab, word2index, index2word = make_indices(vocab)
	print("Fetched and indexed data")
	train_data = convert_to_vector_representation(train_data, word2index)
	valid_data = convert_to_vector_representation(valid_data, word2index)
	print("Vectorized data")
	model = FFNN(input_dim = len(vocab), h = hidden_dim)
	optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9) # This network is trained by traditional (batch) gradient descent; ignore that this says 'SGD'
	print("Training for {} epochs".format(number_of_epochs))
	for epoch in range(number_of_epochs):
		model.train()
		#optimizer.zero_grad()
		loss = 0
		correct = 0
		total = 0
		start_time = time.time()
		print("Training started for epoch {}".format(epoch + 1))
		random.shuffle(train_data) # Good practice to shuffle order of training data
		for input_vector, gold_label in tqdm(train_data):
			predicted_vector = model(input_vector)
			predicted_label = torch.argmax(predicted_vector)
			correct += int(predicted_label == gold_label)
			total += 1
			loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label])) #ERROR 2: += instead of = or indent
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		print("Training completed for epoch {}".format(epoch + 1))
		print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
		print("Training time for this epoch: {}".format(time.time() - start_time))
		model.eval() #added model.eval()
		loss = 0
		correct = 0
		total = 0
		#optimizer.zero_grad() #ERROR 5: Missing zero_grad
		start_time = time.time()
		print("Validation started for epoch {}".format(epoch + 1))
		random.shuffle(valid_data) # Good practice to shuffle order of valid data
		for input_vector, gold_label in valid_data:
			predicted_vector = model(input_vector)
			predicted_label = torch.argmax(predicted_vector)
			correct += int(predicted_label == gold_label)
			total += 1
			loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label])) #ERROR 4: += instead of = or indent
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		print("Validation completed for epoch {}".format(epoch + 1))
		print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
		print("Validation time for this epoch: {}".format(time.time() - start_time))
