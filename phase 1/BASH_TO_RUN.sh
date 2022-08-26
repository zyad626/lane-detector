#!/bin/bash

source "venv/bin/activate"

echo "Enter '-i' for Image mode, '-v' for Video mode:"
read mode
echo "Enter the path (example: test_images/test6.jpg):"
read path

python lane_detector.py $mode $path
