# ongoing: SMT for djk -> zhs 

NOTE: Check the PATH as the folder was moved once

## arpa

data for arpa: https://github.com/christos-c/bible-corpus, generated with kenLM

the data for chinese is tranditional characters, converted to simplified chinese with [langconv](https://pypi.org/project/langconv/)

## command for moses

```
$ $MOSES/scripts/training/train-model.perl -root-dir train -corpus $CORPUS -f djk -e zho -lm 0:5:$PROJECT/smt/lm/bible.blm.zhs -external-bin-dir $MOSES/training-tools/ -parallel --mgiza 
```

## bleu score

- cleaned:

```
$ multi-bleu.perl ../../corpora/zho-djk/test.tk.zho < test.zhs.out
BLEU = 1.98, 18.7/3.6/0.8/0.3 (BP=1.000, ratio=2.189, hyp_len=10653, ref_len=4866)
```

- not cleaned:

```
$ multi-bleu.perl ../../corpora/zho-djk/test.tk.zho < test.zhs.out 
BLEU = 1.93, 19.1/3.7/0.8/0.3 (BP=1.000, ratio=2.191, hyp_len=10659, ref_len=4866)
```
poor result probably due to 

- poor quality of original data (it really looks weird, but i haven't read the chinese version of bible yet, so idk)
- poor alignment (different tokeniser for characters)
- MOSES is not adapted to handle characters (need confirm)
- I am bad at machine translation (most possible reason)
