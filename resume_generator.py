import torch
from cleantext import clean
from os.path import exists
import codecs
import pickle

# x = torch.rand(5, 3)
# print(x)

# y = torch.rand(5, 3)
# print(y)

# print(x+y)

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

	#print (df.head)

	df = df['Resume_str']
	# Alternative
	# df = df['Resume_html']

	return df.values.tolist()

def main():

	file = CLEANED_TEXT_PATH + "/cleaned_text.pickle"

	if not (exists(file)):

		text = get_text_from_csv(RESUME_PATH)
		clean_text(text)

		with open(file, 'wb') as f:
			pickle.dump(text, f)

	with open(file, 'rb') as f:
		testPickle = pickle.load(f)
	
	print(testPickle[1200])

if __name__ == '__main__':
	main()