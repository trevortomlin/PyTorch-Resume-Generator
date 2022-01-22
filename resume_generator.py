import torch
from cleantext import clean
from os.path import exists
import pickle

import numpy as np
import pandas as pd

RESUME_PATH = 'data/Resume/Resume.csv'
CLEANED_TEXT_PATH = 'data/cleaned_text'

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

def main():

	text = get_text_from_csv(RESUME_PATH)
	clean_text(text)

	file = CLEANED_TEXT_PATH + "/cleaned_text.pickle"

	if not (exists(file)):
		write_cleaned_text(text, file)

	cleaned_text = read_cleaned_text(file)
	
	print(cleaned_text[1200])

if __name__ == '__main__':
	main()