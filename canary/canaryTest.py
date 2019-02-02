import unittest
import os
import sys
import time
from canaryLib import canaryLocal
from clint.textui import puts, colored, indent
from nltk.tokenize import sent_tokenize, word_tokenize
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

def canaryTest():
    # Points to the local directory used for the test files
    #directory = ('.././corpus/')
    # Test Array of correct amounts of Argumentative components from local files
    # [filename, major, claim, premise]
    manual = [['essay001', '2', '3', '6'],]
    canary = []
    file = 'essay001.txt'

    canaryLocal(file)
    # [filename, major, claim, premise]
    canary.append([str(file), str(canaryLocal.major), str(canaryLocal.claim), str(canaryLocal.premise)])

    # Need to store the strings instead of just the count, to compare (Will need to manually extract either the count or
    # correct argument components, can use fileCheck to run through every file)
    puts(colored.cyan('---------------------------------------------------'))
    print("Canary vs Manual Analysis: " + "Major Claims: " + canary[0] [1] + "/" + manual[0] [1] + " Claims: "
    + canary[0] [2] + "/" + manual[0] [2] + " Premises: " + canary[0] [3] + "/" + manual[0] [3]) 
    puts(colored.cyan('---------------------------------------------------'))

def CanaryBratAnalysis():
    # Read in .ann files from directory
    directory = ('.././corpus/')
    # Count up components ("MajorClaim", "Claim", "Premise")
    # If a compontent is found +1 to counter
    major_count = 0
    claim_count = 0
    premise_count = 0
    # Used to store the manual analysis via Brat
    manual = []
    
    for files in os.listdir(directory):
        if files.endswith('.ann'):
            f = str(directory+files)
            file = open(f, "r")
            lines = file.readlines()

            for line in lines:
                if 'MajorClaim' in line:
                    major_count += 1
                elif 'Claim' in line:
                    claim_count +=1
                elif 'Premise' in line:
                    premise_count +=1
            # Need to create a var that takes str(files), splits('.')
            manual.append([str(files), str(major_count), str(claim_count), str(premise_count)])

            print(manual)
            time.sleep(5)
                  
if __name__ == '__main__':
    #fileCheck()
    CanaryBratAnalysis()

