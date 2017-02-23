#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 02:28:28 2017

@author: eric
"""
from time import sleep as s


CURSOR_UP="\033[F"
CLEAR_LINE="\033[K"

print("Leave this")
print("This is a test...")
s(1)

""" 
    Go up a line... 
    clear it... 
    then go the beginning of that line and start again
"""
print("{}{}{}".format(CURSOR_UP,CLEAR_LINE,CURSOR_UP))
s(1)
print("DummyScript.com")