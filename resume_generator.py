import torch

# x = torch.rand(5, 3)
# print(x)

# y = torch.rand(5, 3)
# print(y)

# print(x+y)

import numpy as np
import pandas as pd

RESUME_PATH = 'data/Resume/Resume.csv'

def get_text_from_csv(path):

	df = pd.read_csv(path, sep=',', dtype=str, encoding='utf-8', index_col=False)

	df = df['Resume_str']

	return df.values.tolist()

def main():
	
	text = get_text_from_csv(RESUME_PATH)

	#print(text[0])

if __name__ == '__main__':
	main()