#!/bin/bash

BABELNET_VERSION="4.0.1"
BABELNET_FILENAME="BabelNet-API-$BABELNET_VERSION.zip"
BABELNET_URL="https://babelnet.org/data/4.0/$BABELNET_FILENAME"

DOWNLOAD_DESTINATION=/tmp/$BABELNET_FILENAME
UNZIP_PATH=/tmp/$(basename -s .zip $DOWNLOAD_DESTINATION)

LIB_FOLDER="3rdparty/lucene-index"
mkdir -p $LIB_FOLDER/lib

wget $BABELNET_URL -O $DOWNLOAD_DESTINATION
unzip $DOWNLOAD_DESTINATION -d /tmp

cp $UNZIP_PATH/*.jar $LIB_FOLDER/*.jar
cp -r $UNZIP_PATH/lib $LIB_FOLDER/lib
cp -r $UNZIP_PATH/config config
cp -r $UNZIP_PATH/resources resources