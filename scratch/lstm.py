### LSTM sample TF script.

#!/usr/bin/env python
"""Provides A GUI for specific Machine Learning Use-cases.

TF_Curses is a frontend for processing datasets into machine
learning models for use in predictive functions.
"""
__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2017, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.0.1"
__maintainer__ = "Eric Petersen"
__email__ = "ruckusist@alphagriffin.com"
__status__ = "Prototype"

import ag.logging as log
import os, sys, datetime
from os import system as fire
import collections
from time import sleep
import curses
import curses.textpad

log.set(level=4)

def get_time():
    return datetime.datetime.now().replace(microsecond=0).isoformat().replace('T', ' ')

class TestApp(object):
    def __init__(self):
        self.file = os.path.join(os.getcwd(), "sample.txt")

    def main(self):
        start_time = get_time()
        work_dir = os.getcwd()
        log.info("Starting Program - {}".format(start_time))
        log.debug("launch_dir = {}".format(work_dir))
        sample = self.get_text_file(self.file)
        print(sample.text)


    def get_text_file(self, file_):
        if not os.path.isfile(file_):
            log.warn("{} is an invalid path".format(file_))
            return False
        class sample_text(): pass

        sample_text.text = open(file_, "r").read()
        sample_text.len = len(sample_text.text)
        sample_text.chars = sorted(list(set(sample_text.text)))
        sample_text.len_chars = len(sample_text.chars)
        sample_text.textlines = open(file_, "r").readlines()
        sample_text.nwords = 0
        for l in sample_text.textlines:
            w = l.split()
            sample_text.nwords += len(w)
        # print("Sample Text: \n\t{}".format(sample_text))
        return sample_text


class TestWindow(object):
    def __init__(self, app=None):
        self.stdscr = curses.initscr()
        self.stdscr.border(1)
        self.app = app

    def main(self):
        


    def start_win(self):
        begin_x = 20
        begin_y = 7
        height = 5
        width = 40
        self.win = curses.newwin(height, width, begin_y, begin_x)


    def write2stdscr(self, text):
        self.win.addstr(text)

    def close_win(self):
        curses.endwin()


if __name__ == '__main__':
    # As simple as that.
    app = TestWindow()
    try:
        os.system('clear')
        app.main()

    except KeyboardInterrupt:
        os.system('clear')
        print("Ending Test App")
        app.close_win()
        sys.exit()
#/EOF
