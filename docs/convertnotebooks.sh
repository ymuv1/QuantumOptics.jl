#!/bin/bash
SOURCEPATH=../examples/
BUILDPATH=source/examples/

for f in $SOURCEPATH*.ipynb
do
    jupyter nbconvert --to=rst --FilesWriter.build_directory=$BUILDPATH "$f"
done