# utils.py
# Small utility helpers used across scripts.

import time


def copy_file(in_name, line_num, start_line_num=0, shuffle=False):
    """
    Copy a slice of lines from a text file into a new file.

    Note:
        The 'shuffle' argument is kept for compatibility but is not used.
    """
    count = 0
    with open(in_name, "r", encoding="utf-8") as fr, \
         open(f"{in_name}_{start_line_num}_{line_num}", "w", encoding="utf-8") as fw:
        line = fr.readline()
        for _ in range(start_line_num):
            line = fr.readline()

        while line and count <= line_num:
            fw.write(line)
            count += 1
            line = fr.readline()

    print("done")


class Timer:
    """
    Simple wall-clock timer utility.
    """

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