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

def readAnn(file):
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

def canaryBratAnalysis(file):
    """ Used to compare the outputs of Canary with a manually annotated Gold Standard """

    # Loading file into the local version of Canary
    canary = canaryLocal(file)

    # Used to store Argumentative Components
    majorClaims = canary[1]
    claims = canary[0]
    premises = canary[2]
    
    # Reading analysis file and extracting components
    analysis = readAnn("../corpus/essay001.ann")

    # Used to store Argumentative Components from analysis
    majorClaimsAnn = analysis[1]
    claimsAnn = analysis[0]
    premisesAnn = analysis[2]

    # Stores counts used to compare findings
    claimCount = 0
    claimCountAnn = 0

    # Main loop to compare findings of Canary with Gold Standard    
    for claimAnn in claimsAnn:
        # Increment Count for analysis claim
        claimCountAnn += 1
        for claim in claims:
            if claim in claimAnn or claimAnn in claim:
                print("MATCH FOUND!")
                print("Canary: " + claim)
                print("Analysis: " + claimAnn)
                print("\n")
                # Found a match, increment Canary score
                claimCount += 1

    print("Canary vs Brat (Claims): " + str(claimCount) + "/" + str(claimCountAnn))


if __name__ == '__main__':
    """ Used to test the various features of Canary """
    
    canaryBratAnalysis("../corpus/essay001.txt")
    

