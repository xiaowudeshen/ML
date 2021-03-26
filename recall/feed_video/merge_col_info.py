#!/bin/python3
import sys

def readfile(filename):
    with open(filename) as f:
        for line in f:
            line_list = line.strip().split("\t")
            result = ",".join(line_list)
            print(result)

def process():
    filename = sys.argv[1] 
    readfile(filename)

if __name__ == '__main__':
    process()
