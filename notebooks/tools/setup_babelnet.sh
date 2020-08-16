#!/bin/bash

# We expect the dump to be already available at /nfs/knowledge-glue/...

PROGRAM_NAME=$0

function help
{
    echo "Usage: $PROGRAM_NAME from the folder containing tools."
    exit 1
}

# check working directory
CONTAINER_FOLDER=$(dirname `pwd`)
TOOL_FOLDER=$(basename $CONTAINER_FOLDER)

if [[ $TOOL_FOLDER != "tools" ]]; then
    help
fi

# Prepare lucene-index folder
LUCENE="3rdparty/lucene-index"
mkdir -p $LUCENE

BABELNET_VERSION="4.0.1"
BABELNET_FILE="BabelNet-API-$BABELNET_VERSION.zip"
BABELNET_URL="https://babelnet.org/data/4.0/$BABELNET_FILE.zip"

# Download the dataset
wget $BABELNET_URL /tmp/$BABELNET_FILE
unzip /tmp/$BABELNET_FILE
BABELNET_PATH=`basename $BABELNET_FILE .zip`
cp -r $BABELNET_PATH/lib $LUCENE/lib
cp -r $BABELNET_PATH/babelnet-api-$BABELNET_VERSION.jar $LUCENE/
