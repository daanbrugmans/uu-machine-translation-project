#!/usr/bin/env bash
cd /home/yihu2298/zh_back/models/backtranslation/mert-work
/common/student/courses/MT-5LN711-18/tools/MOSES/ubuntu-16.04/bin/extractor --sctype BLEU --scconfig case:true  --scfile run16.scores.dat --ffile run16.features.dat -r /home/yihu2298/zh_back/corpora/backtranslation//dev.tk.cl.zho -n run16.best100.out.gz
