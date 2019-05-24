#!/bin/sh
./learn_extract_micro.py
./evaluate.py -autoenc -train
./evaluate.py -autoenc -freq
