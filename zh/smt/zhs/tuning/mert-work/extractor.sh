#!/usr/bin/env bash
cd /home/yihu2298/5LN718-MT/uu-machine-translation-project/smt_zh/zhs/tuning/mert-work
/common/student/courses/MT-5LN711-18/tools/MOSES/ubuntu-16.04/bin/extractor --sctype BLEU --scconfig case:true  --scfile run13.scores.dat --ffile run13.features.dat -r /home/yihu2298/5LN718-MT/uu-machine-translation-project/smt_zh/corpora/zho-djk/dev.tk.cl.zho -n run13.best100.out.gz
