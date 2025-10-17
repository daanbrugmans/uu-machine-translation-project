import jieba
import sys

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
