import sys
with open("./log/{}.log".format(sys.argv[1]), "r") as fin:
    for line in fin.readlines():
        line_split = line.split(" ")
        if line_split[2] == "[test]":
            print(line, end="")
