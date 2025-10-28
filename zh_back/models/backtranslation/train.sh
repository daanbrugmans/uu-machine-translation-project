#!/bin/bash

# Kinda useless, why not just set it directly as no one gonna run it
if [[ -v "$MOSES" ]]; then
	MOSES_PATH=$MOSES
else
	MOSES_PATH="/common/student/courses/MT-5LN711-18/tools/MOSES/ubuntu-16.04/"
fi

CORPUS_PATH="$HOME/zh_back/corpora/backtranslation/"
LM="$HOME/zh_back/lm/bible.blm.zhs"

# Concatenate data
# cat $HOME/zh_back/corpora/extra_data/bible.tk.cl.zhs >> $CORPUS_PATH/train.tk.cl.zho
# cat $HOME/zh_back/corpora/extra_data/bible.tk.cl.djk >> $CORPUS_PATH/train.tk.cl.djk

if [[ -e train/model/moses.ini ]]; then
	# Turning
	$MOSES_PATH/scripts/training/mert-moses.pl \
		$CORPUS_PATH/dev.tk.cl.djk $CORPUS_PATH/dev.tk.cl.zho \
		$MOSES_PATH/bin/moses train/model/moses.ini 
else
	# Training
	$MOSES_PATH/scripts/training/train-model.perl \
		-corpus $CORPUS_PATH/train.tk.cl -f djk -e zho \
		-root-dir train -lm 0:5:$LM:8 \
		-external-bin-dir $MOSES_PATH/training-tools \
		-parallel -mgiza > training.log 2>&1
	
	# Cpoy turning, too lazy to make it function 
	$MOSES_PATH/scripts/training/mert-moses.pl \
		$CORPUS_PATH/dev.tk.cl.djk $CORPUS_PATH/dev.tk.cl.zho \
		$MOSES_PATH/bin/moses train/model/moses.ini \
		--mertdir $MOSES_PATH/bin > mert.log 2>&1
fi

