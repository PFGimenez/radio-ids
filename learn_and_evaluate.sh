#!/bin/sh
./learn_extract_micro.py
./evaluate_one_probe.py -autoenc -train -cumul -no-time
./evaluate_one_probe.py -autoenc -freq -cumul -no-time
