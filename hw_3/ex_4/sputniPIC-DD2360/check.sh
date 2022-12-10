#!/bin/bash
for FILE in ./data/*.txt; do
    diff $FILE ./data_gt/$(basename "$FILE")
done
