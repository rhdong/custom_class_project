#!/bin/bash

rm -rf build;
mkdir build;
cd build

cmake  -DCMAKE_BUILD_TYPE=Release -Dsm=80 -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.__path__[0])')" ..

make -j
