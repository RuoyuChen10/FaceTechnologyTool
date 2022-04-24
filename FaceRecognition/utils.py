# -*- coding: utf-8 -*-  

"""
Created on 2021/12/26

@author: Ruoyu Chen
"""

import os

class INFO():
    def __init__(self, save_log):
        self.save_log=save_log
        if os.path.exists(save_log):
            os.remove(save_log)

    def __call__(self, string):
        print(string)
        if self.save_log != None:
            with open(self.save_log,"a") as file:
                file.write(string)
                file.write("\n")

def mkdir(name):
    '''
    Create folder
    '''
    isExists=os.path.exists(name)
    if not isExists:
        os.makedirs(name)
    return 0
