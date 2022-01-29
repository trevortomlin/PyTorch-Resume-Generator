import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from cleantext import clean
from os.path import exists
import pickle

import numpy as np
import pandas as pd

RESUME_PATH = 'data/Resume/Resume.csv'
CLEANED_TEXT_PATH = 'data/cleaned_text'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_text(text):

	for x in range(len(text)):

		text[x] = clean(text[x], lower=True, to_ascii=True, fix_unicode=True)

		text[x] = text[x].replace('\n', " ")
		text[x] = text[x].replace("city ,", "city,")
		text[x] = text[x].replace(" - ", ", ")
		text[x] = text[x].replace(" ,", ",")
		text[x] = text[x].replace(" :", ":")

def get_text_from_csv(path):

	df = pd.read_csv(path, sep=',', dtype=str, encoding='utf-8', index_col=False)

	df = df['Resume_str']

	return df.values.tolist()

def write_cleaned_text(text, file):

	with open(file, 'wb') as f:
		pickle.dump(text, f)

def read_cleaned_text(file):

	with open(file, 'rb') as f:
		text = pickle.load(f)

	return text

class ResumeDataset(Dataset):

	def __init__(self, path):

		self.data = []
		self._load(path)

	def _load(self, path):
		file = CLEANED_TEXT_PATH + "/cleaned_text.pickle"

		if not (exists(file)):
			text = get_text_from_csv(path)
			clean_text(text)
			write_cleaned_text(text, file)

		self.data = read_cleaned_text(file)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, i):
		return self.data[i]

class RNN(nn.Module):

	def __init__(self, input_size, hidden_size, output_size, n_layers=1):
		super(RNN, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers

		self.encoder = nn.Embedding(input_size, hidden_size)
		self.rnn = nn.RNN(hidden_size, hidden_size, n_layers)
		self.decoder = nn.Linear(hidden_size, output_size)

	def forward(self, input_t, hidden_t):
		input_t = self.encoder(input_t.view(1, -1))
		output_t, hidden_t = self.gru(input_t.view(1, 1, -1), hidden_t)
		output_t = self.decoder(output_t.view(1, -1))
		return output_t, hidden_t

	def init_hidden(self):
		return torch.zeros(self.n_layers, 1, self.hidden_size)

def main():

	dataset = ResumeDataset(RESUME_PATH)
	dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

	# FOR TRAINING
	# NEEDS TO BE MODIFIED LATER

	# rnn = RNN(64, 64, 64)

	# input_tensor = torch.zeros(1, 1, 1)
	# hidden_tensor = rnn.init_hidden()

	# output, next_hidden = rnn(input_tensor, hidden_tensor)


if __name__ == '__main__':
	main()