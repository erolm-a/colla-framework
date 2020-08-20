#!/bin/bash

# This script sets up the submodules and downloads basic dependencies
# We assume that the working directory is /root/colla-framework
echo "Installing libraries..."
pip3 install -r requirements.txt
echo "Insatlling spacy corpus"
python3 -m spacy download en_core_web_md

# Install the Spylon kernel
pip3 install spylon-kernel
python3 -m spylon_kernel install

# Install the buffer program
cd notebooks
gcc -o /usr/bin/buffer tools/buffer.c -O2 -Wall --std=c11

# This only downloads the jars, not the BabelNet index.
# Refer to README.md for instructions on how to do this.
./setup_babelnet.sh


