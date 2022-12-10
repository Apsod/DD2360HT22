#!/bin/bash
for FILE in ./data/*.vtk; do
    NAME=$(basename "$FILE")
    echo "Checking $NAME"
    diff -u $FILE ./data_gt/$NAME | diffstat -m 
done
