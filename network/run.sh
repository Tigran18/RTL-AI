#!/bin/bash
cd ~/RTL\ AI/network/build || exit
rm -rf *
cmake ..
make
./neural_network