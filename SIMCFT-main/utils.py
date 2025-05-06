import time
import sys
import os


def copy_file(in_name, line_num, start_line_num=0, shuffle=False):
    count = 0
    fr = open(in_name)
    fw = open(f"{in_name}_{start_line_num}_{line_num}", "w")
    line = fr.readline()
    for i in range(start_line_num):
        line = fr.readline()
    while line and count <= line_num:
        fw.write(line)
        count += 1
        line = fr.readline()
    fw.close()
    fr.close()
    print('done')


class Timer:
    def __init__(self):
        self.start = "tik"
        self.bgt = time.time()

    def tik(self, info="tik"):
        self.bgt = time.time()
        self.start = info
        print(f"{info} start")

    def tok(self, info=""):
        if not info:
            info = self.start
        print(f"{info} done, {round(time.time() - self.bgt, 3)}s after {self.start} start")
        return time.time() - self.bgt

    def now(self):
        return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
