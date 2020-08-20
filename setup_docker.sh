#!/bin/bash

# This script sets up the submodules and downloads basic dependencies
# We assume that the working directory is /root/colla-framework
echo "Installing libraries..."
pip3 install -r requirements.txt
echo "Insatlling spacy corpus"
python3 -m spacy download en_core_web_md

# This only downloads the jars, not the BabelNet index.
# Refer to README.md for instructions on how to do this.
cd notebooks
tools/setup_babelnet.sh