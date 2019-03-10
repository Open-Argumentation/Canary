import unittest
import os
import sys
import time
import csv
from canary import canaryLocal

def exportCSV():
    
    # Test data to see if .csv output is correctly setup
    data = [["Essay", "Major Claims", "Claims", "Premises", "Relations"], ["essay001", "1", "2", "2", "1"], ["essay002", "3", "3", "4", "2"]]
    
    # Creating .csv file
    with open("canaryTest.csv", "w") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)
    
    # Closing file
    csvFile.close()
    
    print("File exported: canaryTest.csv")

def readANN(file):
    """ Used to read in a .ann file and extract Argument Components """

    # Used to store Argumentative Components
    majorclaims = []
    claims = []
    premises = []
    components = []

    # Read in .ann file
    with open(file, "r") as annFile:
        lines = annFile.readlines()
    
    # Main loop that checks if any of the components are on a given line in the file
    for line in lines:
        if "MajorClaim" in line:
            # Splits the line by Tab
            major = line.split('\t')
            # Adding to list
            majorclaims.append(str((major[2].split('\n')[0])))
        elif "Claim" in line:
            # Splits the line by Tab
            claim = line.split('\t')
            # Adding to list
            claims.append(str((claim[2].split('\n')[0])))
        elif "Premise" in line:
            # Splits the line by Tab
            premise = line.split('\t')
            # Adding to list
            premises.append(str((premise[2].split('\n')[0])))
    
    # Adding all of the components to a list
    components.append(claims)
    components.append(majorclaims)
    components.append(premises)

    return components


if __name__ == '__main__':
    """ Used to test the various features of Canary """
    
    brat = readANN("../corpus/essay001.ann")
    print(str(brat))
    

