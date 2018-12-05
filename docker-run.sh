#!/bin/bash 

docker logs -f $(docker run --rm -d --name server2  -v /home/sahand/Projects/Go/src/github.com/sahandhnj/ml-deployment-benchmarks/meta:/runtime/meta -p 3002:3002 coco:latest)