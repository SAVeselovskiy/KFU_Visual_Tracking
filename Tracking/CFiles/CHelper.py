__author__ = 'sergejveselovskij'
from ctypes import cdll
import ctypes
lib = cdll.LoadLibrary('./CFiles/libchelper.so')

class CHelper(object):
    def __init__(self):
        self.obj = lib.helper()

    def getHelpNumber(self, string):
        lib.getHelpNumber.restype = ctypes.c_float
        lib.getHelpNumber.argtypes = ()
        return lib.getHelpNumber(self.obj, ctypes.c_char_p(string))
