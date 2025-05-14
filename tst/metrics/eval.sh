#!/bin/bash

# For en change the below if-else dep on whether you want sentence or para leevel
LANG="vi"
if [ "$LANG" = "en" ]; then
   #SUBSET="default"
   SUBSET="paras"
elif [ "$LANG" = "pt" ]; then
   SUBSET="default"
elif [ "$LANG" = "vi" ]; then
   SUBSET="mpairs"
fi
PROMPT_TECHNIQUES=("minimal" "instruct" "few1" "few2" "few3" "few4" "few5")

python tst/metrics/eval.py --lang $LANG --subset $SUBSET --prompt_techs "${PROMPT_TECHNIQUES[@]}"
