import abc
import warnings
import numpy as np
class Matcher(abc.ABC):

    @abc.abstractmethod
    def initialize(self, to, what):
        pass
    @abc.abstractmethod
    def __init__(self, match, mu, delta):
        pass

    @abc.abstractmethod
    def initializeMatrices(self, img1, img2):
        pass

    @abc.abstractmethod
    def calculateMatrices(self, to, what, currentIndex):
        pass

