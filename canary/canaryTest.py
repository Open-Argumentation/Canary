import unittest
import os
import sys
import time
from canaryLib import canaryLocal
from clint.textui import puts, colored, indent
from nltk.tokenize import sent_tokenize, word_tokenize

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
    # Used to store the results from Canary
    canary = []
    f = 0
    for files in os.listdir(directory):
        if files.endswith('.ann'):
            input = str(directory+files)
            file = open(input, 'r')
            filename = os.path.splitext(files)
            lines = file.readlines()

            for line in lines:
                if 'MajorClaim' in line:
                    major_count += 1
                elif 'Claim' in line:
                    claim_count +=1
                elif 'Premise' in line:
                    premise_count +=1
            # Need to create a var that takes str(files), splits('.')
            manual.append([str(filename[0]), str(major_count), str(claim_count), str(premise_count)])
            
            # Now need to read in the same file into Canary and compare results
            extension = '.txt'
            canaryLocal(str(filename[0]) + extension)
            # [filename, major, claim, premise]
            canary.append([str(filename[0]), str(canaryLocal.major), str(canaryLocal.claim), str(canaryLocal.premise)])
                  
            print("Canary vs Manual Analysis [" + str(filename[0]) + "]: " + "Major Claims: " + canary[f] [1] + "/" + manual[f] [1] + " Claims: "
            + canary[f] [2] + "/" + manual[f] [2] + " Premises: " + canary[f] [3] + "/" + manual[f] [3])
            puts(colored.cyan('---------------------------------------------------'))
               
            time.sleep(3)
            f+=1

if __name__ == '__main__':
    #fileCheck()
    CanaryBratAnalysis()

