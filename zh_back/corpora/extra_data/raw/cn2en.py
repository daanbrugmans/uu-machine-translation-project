import torch
from transformers import pipeline

print('CUDA Test:', torch.cuda.is_available())

pipeline = pipeline(
    task="translation",
    model="facebook/mbart-large-50-many-to-many-mmt",
    device=0,
    dtype=torch.float16,
    src_lang="zh_CN",
    tgt_lang="en_XX",
)

with open('bible.zhs', 'r') as f:
    sentences = f.readlines()

# Tested: keep spaces is fine
sentences = [sent.strip() for sent in sentences]
translations = pipeline(sentences)
translation = [i['translation_text'] for i in translations]

with open('bible.eng', 'w') as f:
    f.write('\n'.join(translation))

