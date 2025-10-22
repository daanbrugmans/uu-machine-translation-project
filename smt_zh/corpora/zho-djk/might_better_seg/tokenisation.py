'''
argv[1]: read
argv[2]: write
argv[3]: (opt) dictionary
'''

import thulac
import sys

try:
    thu1 = thulac.thulac(user_dict=sys.argv[3], seg_only=True)
except IndexError:
    thu1 = thulac.thulac(seg_only=True)

thu1.cut_f(sys.argv[1], sys.argv[2])

'''
def tokenise(sent: str) -> str:
    sent = sent.strip()    
    seg_list = thu1.cut(sent)

    return ' '.join(seg_list)

with open(sys.argv[1], 'r') as f:
    sents = f.readlines()

sents = [tokenise(sent) for sent in sents]

with open(sys.argv[2], 'w') as f:
    for sent in sents:
        f.write(sent + '\n')
'''

