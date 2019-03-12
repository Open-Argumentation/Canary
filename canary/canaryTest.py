import unittest
import os
import sys
import time
import csv
import time
from os.path import join
from glob import glob
from canary import canaryLocal

def exportCSV(data):
    
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

def canaryBratAnalysis(fileTxt, fileAnn):
    """ Used to compare the outputs of Canary with a manually annotated Gold Standard """

    # Directory
    directory = "../corpus/"

    # Loading file into the local version of Canary
    canary = canaryLocal(directory + fileTxt)

    # Used to store Argumentative Components
    majorClaims = canary[1]
    claims = canary[0]
    premises = canary[2]
    
    # Reading analysis file and extracting components
    analysis = readAnn(directory + fileAnn)

    # Used to store Argumentative Components from analysis
    majorClaimsAnn = analysis[1]
    claimsAnn = analysis[0]
    premisesAnn = analysis[2]

    # Stores counts used to compare findings
    majorCount = 0
    majorCountAnn = 0
    claimCount = 0
    claimCountAnn = 0
    premiseCount = 0
    premiseCountAnn = 0
    

    # Main loop to compare findings of Canary with Gold Standard    
    for majorClaimAnn in majorClaimsAnn:
        majorCountAnn += 1
        for majorClaim in majorClaims:
            if majorClaim.lower() in majorClaimAnn.lower() or majorClaimAnn.lower() in majorClaim.lower():
                # Found a match, increment Canary score
                majorCount += 1
                
    for claimAnn in claimsAnn:
        # Increment Count for analysis claim
        claimCountAnn += 1
        for claim in claims:
            if claim.lower() in claimAnn.lower() or claimAnn.lower() in claim.lower():
                # Found a match, increment Canary score
                claimCount += 1

    for premiseAnn in premisesAnn:
        premiseCountAnn += 1
        for premise in premises:
            if premise.lower() in premiseAnn.lower() or premiseAnn.lower() in premise.lower():
                # Found a match, increment Canary score
                premiseCount += 1

    print("Canary vs Brat (Major Claims): " + str(majorCount) + "/" + str(majorCountAnn))
    print("Canary vs Brat (Claims): " + str(claimCount) + "/" + str(claimCountAnn))
    print("Canary vs Brat (Premise): " + str(premiseCount) + "/" + str(premiseCountAnn))

    # Stores all the counts
    data = [["Essay", "Method", "Major Claims", "Claims", "Premises", "Relations"]]
    data.append([fileTxt, "Canary", str(majorCount), str(claimCount), str(premiseCount), "Relations"])
    data.append([fileAnn, "Gold Standard", str(majorCountAnn), str(claimCountAnn), str(premiseCountAnn), "Relations"])
    
    return data

def canaryComponentTest(directory):
    """ Main testing function to compare the results of Canary with the Gold Standard """

    # Stores what type of files we are looking for in the directory
    types = ("*txt", "*.ann")

    # Stores the files that match those types (same filename has both .txt & .ann)
    files = []

    for extension in types:
        files.extend(glob(join(directory, extension)))

    for file in files:
        # Spliting the filename from directory
        filename = (file.split(directory))
        # Filename with no extension (.txt, .ann)
        filename = (filename[1].split(".")[0])
        # Comparing file results (Canary vs "Gold Standard")
        analysis = canaryBratAnalysis(filename + ".txt", filename + ".ann")
        # Exporting results to .csv file
        exportCSV(analysis)


if __name__ == '__main__':
    """ Used to test the various features of Canary """
    canaryComponentTest("../corpus/")

