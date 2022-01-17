import numpy as np
import pandas as pd

RESUME_PATH = 'data/Resume/Resume.csv'

def main():
	
	df = pd.read_csv(RESUME_PATH, sep=',', dtype=str, encoding='utf-8', index_col=False)

	df.drop('Resume_html', axis=1, inplace=True)
	df.drop('ID', axis=1, inplace=True)

	print(df.head)

if __name__ == '__main__':
	main()