'''
argv[1]: read
argv[2]: write
argv[3]: (opt) dictionary
'''

import jieba
import sys

try:
    jieba.load_userdict(sys.argv[3])
except IndexError:
    pass

def tokenise(sent: str) -> str:
    sent = sent.strip()    
    seg_list = jieba.cut(sent)

    return ' '.join(seg_list)

with open(sys.argv[1], 'r') as f:
    sents = f.readlines()

sents = [tokenise(sent) for sent in sents]

with open(sys.argv[2], 'w') as f:
    for sent in sents:
        f.write(sent + '\n')
