#!/bin/sh
# Date: 2013-11-7
# Author: shen zhun

echo caculate word frequency
tr -sc '[A-Z][a-z]' '[\012*]' < test | sort | uniq -c | sort -n > test.fq
echo check the first 80 word frequency
cat test.fq | tail -80
