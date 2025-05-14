#!/bin/bash

LANG="en"
DS="default"
python tst/data/gen_eval_set.py $LANG $DS

LANG="en"
DS="paras"
python tst/data/gen_eval_set.py $LANG $DS

LANG="pt"
DS="default"
python tst/data/gen_eval_set.py $LANG $DS

LANG="vi"
DS="mpairs"
python tst/data/gen_eval_set.py $LANG $DS