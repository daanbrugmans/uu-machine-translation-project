#!/usr/bin/env bash
cd /home/yihu2298/zh_back/models/eng-djk/mert-work
/common/student/courses/MT-5LN711-18/tools/MOSES/ubuntu-16.04/bin/extractor --sctype BLEU --scconfig case:true  --scfile run6.scores.dat --ffile run6.features.dat -r /home/yihu2298/zh_back/corpora/eng-djk/dev.tk.lc.clean.djk -n run6.best100.out.gz
