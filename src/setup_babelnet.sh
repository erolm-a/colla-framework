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

cp -r $UNZIP_PATH/babelnet-api-$BABELNET_VERSION.jar $LIB_FOLDER/
cp -r $UNZIP_PATH/lib/*.jar "$LIB_FOLDER/lib"
cp -r $UNZIP_PATH/config config
cp -r $UNZIP_PATH/resources resources

# Compile our BabelNet bridge
mkdir tools/BabelNet/lib
cp $LIB_FOLDER/*.jar tools/BabelNet/lib/
cp $LIB_FOLDER/lib/*.jar tools/BabelNet/lib/
cd tools/BabelNet
sbt compile && sbt package
