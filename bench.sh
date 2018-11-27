#!/bin/bash

venv/bin/locust -f locust/locustfile.py --host=http://localhost:3002/imagedetector/v1