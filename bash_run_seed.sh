#!/bin/bash

for (( i=1 ; i<100 ; i+=1 ))
do
    python eval_byot_svm_abide.py -s $i
done
