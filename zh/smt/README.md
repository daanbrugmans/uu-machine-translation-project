# ongoing: SMT for djk -> zhs 

TODO: waiting for backtranslation, don't think there would be anything else for smt

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

- might-better-segmentation:

i tried 

```
$ multi-bleu.perl ../../corpora/zho-djk/might_better_seg/test.tk.zho < test.out.zho 
BLEU = 0.89, 13.4/2.0/0.4/0.1 (BP=1.000, ratio=2.545, hyp_len=10766, ref_len=4231)
```
- cleaned + tuning

```
$ multi-bleu.perl ../../corpora/zho-djk/test.tk.zho < test.out.zhs                  
BLEU = 2.17, 19.4/3.8/0.9/0.3 (BP=1.000, ratio=2.034, hyp_len=9899, ref_len=4866)

$ cat mert-work/weights.txt 
0.0579334 0.393204 0.0601482 0.240538 0.0845546 0.127658 0.0127795 -0.0231838 
```

tuning with validation set of 111 rows data, default parameters

---

poor result probably due to 

- poor quality of original data (it really looks weird, but i haven't read the chinese version of bible yet, so idk)
- poor alignment (different tokeniser for characters)
- MOSES is not adapted to handle characters (need confirm)
- I am bad at machine translation (most possible reason)
