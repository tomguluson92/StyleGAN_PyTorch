from torchvision_sunner.constant import *

"""
    This script defines the function which are widely used in the whole package

    Author: SunnerLi
"""

def quiet():
    """
        Mute the information toward the whole log in the toolkit
    """
    global verbose
    verbose = False

def INFO(string = None):
    """
        Print the information with prefix

        Arg:    string  - The string you want to print
    """
    if verbose:
        if string:
            print("[ Torchvision_sunner ] %s" % (string))
        else:
            print("[ Torchvision_sunner ] " + '=' * 50)