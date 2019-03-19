import unittest
import os
import sys
import time
import csv
import time
from os.path import join
from glob import glob
from canary import Local, Relations

def exportCSV(data):
    """ Exporting data from Canary to a .csv file for inspectation/graphing """
    # Creating .csv file
    with open("canaryTest.csv", "a") as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)
    
    # Closing file
    csvFile.close()
    
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
    components.append(majorclaims)
    components.append(claims)
    components.append(premises)

    return components

def readAnnRelations(file):
    """ Used to read in a .ann file and extract Argument Components """

    # Used to store Argumentative Components
    components = []
    relations = []
    links = []
    
    # Read in .ann file
    with open(file, "r") as annFile:
        lines = annFile.readlines()
    
    # Main loop that checks if any of the components are on a given line in the file
    for line in lines:
        # Splits the line by Tab
        test = line.split('\t')
        # Components are printed here by: "T1" etc
        if "T" in test[0]:
            # Adding Component ID [0] and Component [1] to list
            components.append([test[0], test[2].split('\n')[0]])
        elif "R" in test[0]:
            # Premise = [1], Claim/Major = [2]
            support = (test[1].split("Arg")[1]).split(":")[1]
            claim = (test[1].split("Arg")[2]).split(":")[1]
            relations.append([claim, support])
    
    for relation in relations:
        #print("Relation: " + str(relation))
        for component in components:
            if relation[0] in component:
                claim = component[1]
            elif relation[1].split(" ")[0] in component:
                support = component[1]
    
    for relation in relations:
        for component in components:
            if relation[0] in component[0]:
                claim = component[1]
            if relation[1].split(" ")[0] in component[0]:
                support = component[1]
        # Adding "linked" components to a list so I can compare
        links.append([claim, support])
    
    # Adding all of the lniked components to a list
    
    return links

def BratAnalysis(fileTxt, fileAnn):
    """ Used to compare the outputs of Canary with a manually annotated Gold Standard """

    # Directory
    directory = "../corpus/"

    # Loading file into the local version of Canary
    canary = Local(directory + fileTxt)

    # Used to store Argumentative Components
    majorClaims = canary[0]
    claims = canary[1]
    premises = canary[2]
    
    # Reading analysis file and extracting components
    analysis = readAnn(directory + fileAnn)

    # Used to store Argumentative Components from analysis
    majorClaimsAnn = analysis[0]
    claimsAnn = analysis[1]
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

    # Stores all the counts
    counts = [[majorCount, claimCount, premiseCount], [majorCountAnn, claimCountAnn, premiseCountAnn]]
    
    return counts

def BratRelationAnalysis(fileTxt, fileAnn):
    """ Used to compare relation results """
    
    # Directory
    directory = "../corpus/"

    # Loading file into the local version of Canary
    canary = Local(directory + fileTxt)

    # Used to store Argumentative Components
    claims = canary[1]
    premises = canary[2]

    # Not really needed, I could loop Canary[1]/Canary[2]
    # Finding relations via canaryRelations
    canary = Relations(claims, premises)

    relations = []

    for components in canary:
        # Claim, Premise
        relations.append([components[0], components[1]])
    
    # Reading analysis file and extracting components
    analysisRelations = readAnnRelations(directory + fileAnn)

    # Stores counts used to compare findings
    relationsCount = 0
    analysisRelationsCount = 0

    for relation in relations:
        for analysisRelation in analysisRelations:
            if relation[0].lower() in analysisRelation[0].lower() or analysisRelation[0].lower() in relation[0].lower():
                if relation[1].lower() in analysisRelation[1].lower() or analysisRelation[1].lower() in relation[1].lower():
                    relationsCount += 1

    # Working out count for Gold Standard
    for analysisRelation in analysisRelations:
        analysisRelationsCount+= 1
    
    # Stores counts
    counts = [[relationsCount, analysisRelationsCount]]

    return counts

def Test(directory):
    """ Main testing function """
    """ Testing function to compare relation results of Canary vs the Gold Standard """

    # Stores what type of files we are looking for in the directory
    types = ("*txt", "*.ann")

    # Stores the files that match those types (same filename has both .txt & .ann)
    files = []

    for extension in types:
        files.extend(glob(join(directory, extension)))

    for file in files:
        # Printing filename for testing (Canary Relations breaks on something)
        print("Incase Break: " + file)
        # Spliting the filename from directory
        filename = (file.split(directory))
        # Filename with no extension (.txt, .ann)
        filename = (filename[1].split(".")[0])
        # Comparing Components results (Canary vs "Gold Standard")
        componentsAnalysis = BratAnalysis(filename + ".txt", filename + ".ann")
        # Comparing Relations results (Canary vs "Gold Standard")
        relationsAnalysis = BratRelationAnalysis(filename + ".txt", filename + ".ann")
        # Exporting results to .csv file
        data = []
        data.append([filename, "Canary", str(componentsAnalysis[0][0]), str(componentsAnalysis[0][1]), str(componentsAnalysis[0][2]), str(relationsAnalysis[0][0])])
        data.append([filename, "Manual", str(componentsAnalysis[1][0]), str(componentsAnalysis[1][1]), str(componentsAnalysis[1][2]), str(relationsAnalysis[0][1])])
        # Add another line for f1-score for components/relations
        exportCSV(data)
        print("File: " + filename + " exported to canaryTest.csv")
            
if __name__ == '__main__':
    """ Used to test the various features of Canary """
    Test("../corpus/")