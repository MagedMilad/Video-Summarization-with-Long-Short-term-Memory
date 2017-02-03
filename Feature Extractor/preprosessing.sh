#!/bin/bash
# FILES is the folder of the database, it's a folder of folders of images
FILES=/home/magedmilad/GP/test/*
for f in $FILES
do
   python googlenet.py $f
done