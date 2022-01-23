#from math import *
from cmath import  pi
import matplotlib.pyplot as mpl
from numpy import arange, exp, sqrt
from typing import Final

class ComplexExponential(object):
    def __init__(self, k, N, Normalized = False):
        assert N > 0 & isinstance(N, int), "N must be a non-negative scalar"
        self.k : Final= k
        self.N : Final = N
        self.ns : Final = arange(N) # Alternate: terms = arange(N)

        self.e_kN : Final = exp(2j * pi * k * self.ns / N) / (sqrt(N) if not Normalized else 1)


        # apparently these names have to mirror e_kN? Might be a jypyter thing, idfk. f* numpy rn
        # Vector containing real elements of the complex exponential
        #self.e_kN_real = self.e_kN.real

        # Vector containing imaginary elements of the complex exponential
        #self.e_kN_imag = self.e_kN.imag

    def plot(self):
        raise Exception("TODO")
