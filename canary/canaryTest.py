import unittest
import os
import sys
import time
from canaryLib import canaryLocal
from clint.textui import puts, colored, indent
class CanaryTest(unittest.TestCase):

    def test(self):
        self.assertEqual('1', '1')


def fileCheck():
    directory = ('.././corpus/')
    files = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            # Adding all of the .txt files into a list
            files.append(filename)
            # Run Canary.local on the file
            canaryLocal(filename)
            # Using Clint to break up outputs
            puts(colored.cyan('---------------------------------------------------'))
            # Easier to take in the application running... (with a wait)
            time.sleep(3)

    
if __name__ == '__main__':
    fileCheck()

