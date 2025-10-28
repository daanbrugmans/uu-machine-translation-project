#!/bin/bash

# Kinda useless, why not just set it directly as no one gonna run it
if [[ -v "$MOSES" ]]; then
	MOSES_PATH=$MOSES
else
	MOSES_PATH="/common/student/courses/MT-5LN711-18/tools/MOSES/ubuntu-16.04/"
fi

RAW_ENG_DATA="raw/bible.eng"
RAW_ZHS_DATA="raw/bible.zhs"
LM="$HOME/zh_back/lm/all.blm.djk"
MODEL_PATH="$HOME/zh_back/models/eng-djk"

# Tokeniser
$MOSES_PATH/scripts/tokenizer/tokenizer.perl -l en < $RAW_ENG_DATA > raw/bible.tk.notlower.eng

# Lower case
$MOSES_PATH/scripts/tokenizer/lowercase.perl < raw/bible.tk.notlower.eng > bible.tk.eng

# Cleaning data
cp $RAW_ZHS_DATA bible.tk.zhs
$MOSES_PATH/scripts/training/clean-corpus-n.perl bible.tk eng zhs bible.tk.cl 1 80

# Get translation (eng -> djk)
# The output should be tokenised and lowercased 
$MOSES_PATH/bin/moses -f $MODEL_PATH/mert-work/moses.ini < bible.tk.cl.eng > bible.tk.cl.djk
