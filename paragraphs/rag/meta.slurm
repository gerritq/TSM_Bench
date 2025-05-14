#!/bin/bash

# rag-search-1
# API_KEY=""
# CX_ID=""

# rag-search-2
API_KEY=""
CX_ID=""

LANG='pt'
SUBSET='extend'
N=2700

python meta2.py $LANG $SUBSET $N $API_KEY $CX_ID
