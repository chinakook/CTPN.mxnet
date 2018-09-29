#!/bin/bash
mkdir -p ./model
cd rcnn/cython
python setup.py build_ext --inplace
cd ../..
