#!/usr/bin/env python3
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from termcolor import cprint
import tensorflow as tf

import lib.options as opts
import lib.prepare as prep



# Play
class webServer(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        output = [
            int(1),
            int(2),
            int(3),
            int(4),
            int(5),
        ]

        cprint("AI: " + str(output), 'green')

        ### respond with action
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(output)
        return

    
class Play(object):
    def __init__(self,options):
        self.options = options
        # need some pathy kind of stuff here
        self.save_path = self.options.save_dir + '_best_validation_1_'
    
    def load_graph(self, session):
        session = tf.Session
        saver = tf.train.Saver()
        
        # this path here will be passed by the selectorator
        # THERE ARE NO FILE EXTIONONS IN THE FUTURE!!!!
        save_path = self.options.log_dir + 'alpha.griffin'
        saver.restore(sess=session, save_path=save_path)
            
    def classify(self, Image):
        img = prepare_image(Image)
        joystick = _best_validation_1_
        output = [
                int(joystick[0] * 80),
                int(joystick[1] * 80),
                int(round(joystick[2])),
                int(round(joystick[3])),
                int(round(joystick[4])),
            ]
        
if __name__ == '__main__':
    options = opts.options(verbose=False)
    blank_string = ''  # wtf over?
    server = HTTPServer((blank_string, options.port), webServer)
    
    
    classifier = Play()
    print ('Started httpserver on port ' , PORT_NUMBER)
    server.serve_forever()        
