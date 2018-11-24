#!/bin/bash 

cd v3
go build -o server
ls
mv server ../server
cd ..
rm output/*
rm input/*
./server