#!/usr/bin/env bash

# Runs train.py and saves the console output to output/out
print("start")
stdbuf -i0 -o0 -e0 python3 -u train.py | tee output/out
print("end")