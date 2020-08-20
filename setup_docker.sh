#!/bin/bash

# This script sets up the submodules and downloads basic dependencies
# We assume that the working directory is /root/colla-framework
echo "Installing libraries..."
pip3 install -r requirements.txt
echo "Insatlling spacy corpus"
python3 -m spacy download en_core_web_md

# FIXME: sometimes it is necessary to re-run this command as kernels do not get
# properly initialized before jupyter-lab is run for the first time
pip3 install spylon-kernel
python3 -m spylon_kernel install

# This only downloads the jars, not the BabelNet index.
# Refer to README.md for instructions on how to do this.
cd notebooks
./setup_babelnet.sh

# Compile our BabelNet bridge
mkdir tools/BabelNet/lib
cp 3rdparty/lucene-index/*.jar tools/BabelNet/lib/
cp 3rdparty/lucene-index/lib/*.jar tools/BabelNet/lib/
cd tools/BabelNet
sbt compile && sbt package
