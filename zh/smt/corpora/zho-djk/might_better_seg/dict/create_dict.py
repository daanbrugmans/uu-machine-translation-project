import sys
import pandas as pd

with open(sys.argv[1], 'r') as f:
    sents = f.readlines()

sents = [sent.strip().split() for sent in sents]

words = []
for sent in sents:
    words.extend(sent)

count = pd.Series(words).value_counts().to_frame()
count.to_csv('dict.csv', sep=' ', header=False)
