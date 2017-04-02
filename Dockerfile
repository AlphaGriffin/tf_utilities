FROM python:3.6.1-alpine

WORKDIR /

RUN apk update && apk upgrade && \
    apk add --no-cache bash git openssh

# setup dependcies not in pip
RUN git clone https://github.com/alphagriffin/logpy
WORKDIR /logpy
RUN python3 setup.py install
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ end basic setup
WORKDIR /
RUN git clone https://github.com/ruckusist/TF_Curses
WORKDIR /TF_Curses
RUN python3 setup.py install

ENTRYPOINT tf_curses