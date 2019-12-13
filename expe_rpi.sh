#!/bin/sh
rm test-rpi/cnn-raspi3-results-autoenc.joblib test-rpi/results*
./evaluate_one_probe.py -autoenc -freq -cumul -no-time
