# Copyright (C) 2017 Alpha Griffin
# @%@~LICENSE~@%@

"""
TF_Curses
Alphagriffin.com
Eric Petersen @Ruckusist <eric.alphagriffin@gmail.com>
"""

__author__ = "Eric Petersen @Ruckusist"
__copyright__ = "Copyright 2017, The Alpha Griffin Project"
__credits__ = ["Eric Petersen", "Shawn Wilson", "@alphagriffin"]
__license__ = "***"
__version__ = "0.0.1"
__maintainer__ = "Eric Petersen"
__email__ = "ruckusist@alphagriffin.com"
__status__ = "Prototype"

import os
import datetime.datetime as dt

class chatbot(object):
    def __init__(self, user=None, Session=None):
        if user is None:
            user = 'Prof_Falken'
        self.current_user = user
        if Session is None:
            Session = 420
        self.current_sess = Session
        self.current_dialog = '{}-{}-dialog.txt'.format(Session, user)
        save_file = os.path.join(os.getcwd(), self.current_dialog)
        _ = self.concat("--$ Starting a conversation with {}".format(user), echo=True)
        # start tensorflow
        # load model
        # test script with first word for + 1 word for greeting
        # all good boolean

    def talk(self, message):
        """private method"""
        # add the message to our compete conversation
        flags = self.chat_input(message)
        Issues = []
        if flags:
            Issues.append(self.dealwith(flag) for flag in flags)
        conversation = self.concat(issues=Issues, message=message)
        current_thought = self.chat_output(conversation)
        return current_thought

    def chat_input(self, message):
        """Filters the chat for keywords. can you do it dynamically?"""
        flags = []
        if 'services' in message:
            flags.append(['services'])
        return flags

    def dealwith(self, flag):
        issues = []
        results = self.search(flag)
        if results:
            for key in results:
                value = self.load_data(key)
                issues.append([key, value])
            return issues
        else:
            key = str(flag)
            value = 'flag'
            self.store_data([key, value])
            issues.append([key, value])
            return issues

    def search(self, flag):
        result = 0
        return result

    def concat(self, message, issues=None, echo=False):
        conversation = None
        with open(self.current_dialog, 'w', encoding='utf-8') as current_text:
            if echo:
                current_text.write("chatbot|{}|: {}".format(datetime.now().isoformat(timespec='minutes'), message))
            else:
                current_text.write("{}|{}|: {}".format(self.current_user, datetime.now().isoformat(timespec='minutes'), message))
            if issues:
                current_text.write("Current Issues: {}".format(issues))
            conversation = current_text.read()
        return conversation

    def store_data(self, key, value):
        # save to database
        x = [(key, value)]
        del x
        return True

    def load_data(self, key):
        # load data from database ...
        value = "key = {}".format(key)
        return key, value

    def chat_output(self, conversation):
        # log this instance

        # LOGICS ---
        # feel the tensors flow
        message = "{}".format(message[::1])
        _ = self.concat(message, echo=True)
        # /LOGICS --
        # given a return
        return message