#!/bin/sh
# Rebuild the results and the website data from the extracted features.
python3 train2.py && python3 calibrate.py && python3 finalize.py
