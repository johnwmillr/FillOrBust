# MussBuss

import numpy as np


class Turn(object):

    def __init__(self, score=0, birthright=0):
        self.score = score
        self.birthright = birthright


class Player(object):

    def __init__(self, name='', score=0):
        self.score =0
        self.name = name


class Die(object):

    def __init__(self, val):
        self.val = val
        self._held = False

    # def isPointer(self):
    #     return self.val in
