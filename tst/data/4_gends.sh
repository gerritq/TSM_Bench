#!/bin/bash

# We experiemtn with different dataset versions. We use the following for the final paper

# en
# LANG="en"
# DS="default"

# en (for paragraphs only)
LANG="en"
DS="mpairs"

# pt
# LANG="pt"
# DS="mpairs"

# vi
LANG="vi"
DS="mpairs"


python3 tst/data/4_gends.py $LANG $DS
