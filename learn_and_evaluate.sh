#!/bin/sh
./learn_extract_micro.py
./evaluate_one_probe.py -autoenc -train
./evaluate_one_probe.py -autoenc -freq
