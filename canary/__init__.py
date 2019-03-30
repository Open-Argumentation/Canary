# Imports
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim import downloader as data
from os.path import join
from glob import glob
import SADFace as sf
import json
import sys
import os
import random
import csv
import time
import unittest

def Preprocessing(file, type):
    """ Used to pre-process an input file to order to efficient produce results. """
    
    # Pre-processes an Input file
    if type == "text":
        # User file imported, converted to lowercase and case ignored (NEED TO ADD CASE IGNORE)
        userFile = open(file).read().lower()
        # Tokenizing userFile into sentences, also removes Whitespace/Breaks
        sentenceTokens = sent_tokenize(userFile)
        return sentenceTokens 
    # Pre-procosses an Argumentative Component
    elif type == "component":
        # Reading in stopWords.json
        with open("data/stopwords.json") as stopWordsFile:
            sWords = json.load(stopWordsFile)

             # Used to store Stopwords
            stopWords = []

            # Looping through JSON file to add to list
            for i in xrange(0, len(sWords["stopwords"])):
                stopWords.append(str(sWords["stopwords"][i]))

        # Tokenization (should switch to wordtokenizer by NLTK?)
        tokens = file.lower().split()
        # Removing Stop words
        tokens = [w for w in tokens if w not in stopWords]
        return tokens
    else:
        print("SEE DOCUMENTATION FOR CORRECT USAGES")


def Local(file):
    """ Finds Argumentative Components in a local file """

    # Store Components
    claim = []
    majorClaim = []
    premise = []
    components = []

    # Importing Indicators via "indicators.json"
    with open("data/indicators.json") as indicatorsFile:
        indicators = json.load(indicatorsFile)

        # Store Indicators in their respective lists
        claimIndicators = []
        majorClaimIndicators = []
        premiseIndicators = []

        # Looping through JSON file to add Indicators to their respective lists
        for i in xrange(0, len(indicators["indicators"])):
            # Claim
            for i in range(len(indicators["indicators"][i]["claim"])):
                claimIndicators.append(str(indicators["indicators"][0]["claim"][i]))
            i = 0
            # Major
            for i in range(len(indicators["indicators"][i]["major"])):
                majorClaimIndicators.append(str(indicators["indicators"][0]["major"][i]))
            i = 0
            # Premise
            for i in range(len(indicators["indicators"][i]["premise"])):
                premiseIndicators.append(str(indicators["indicators"][0]["premise"][i]))
            i = 0
        
        # Importing and pre-processing the User's input file
        sentenceTokens = Preprocessing(file, "text")

        # Looping through userFile Tokens (sentences)
        for line in xrange(0, len(sentenceTokens)):
            # Claim Indicators loop
            for i in range(len(claimIndicators)):
                if (" " + claimIndicators[i] + " ") in (" " + sentenceTokens[line] + " "):
                    # Store current Component
                    claimComponent = str(sentenceTokens[line])
                    # Check to see if Component is already in list
                    if claimComponent not in str(claim):
                        # Add to found claims
                        claim.append(claimComponent)

            # Major Indicators loop
            for i in range(len(majorClaimIndicators)):
                # Indicator found in a given sentence
                if (" " + majorClaimIndicators[i] + " ") in (" " + sentenceTokens[line] + " "):
                    # Store current Component
                    claimMajorComponent = str(sentenceTokens[line])
                    # Check to see if Component is already in list
                    if claimMajorComponent not in str(majorClaim):
                        # Add to found claims
                        majorClaim.append(claimMajorComponent)

            # Premise Indicators loop
            for i in range(len(premiseIndicators)):
                # Indicator found in a given sentence
                if (" " + premiseIndicators[i] + " ") in (" " + sentenceTokens[line] + " "):
                    # Store current Component
                    premiseComponent = str(sentenceTokens[line])
                    # Check to see if Component is already in list
                    if premiseComponent not in str(premise):
                        # Add to found claims
                        premise.append(premiseComponent)

    # All components add to a list to be returned/re-used in other functions
    components.append(majorClaim)
    components.append(claim)
    components.append(premise)
    return components

def Relations(claims, premises):
    """ Finds Argumentative Relations from a list of Claims/Premises """

    # Store Relations
    relations = []
    # Stores used premises
    usedPremises = []
    # Stores used claims
    leftoverPremises = []
    
    # Inputting pre-trained data from Wikipedia 2014+ (word-vectors)
    wordVectors = data.load("glove-wiki-gigaword-100")
    
    # Attempt Three
    for claim in claims:
        # Pre-processing each claim in order to efficiently compare it against a premise
        claimTokens = Preprocessing(claim, "component")
        # Stores comparisons between a given premise and claims
        comparisons = []
        for premise in premises:
            if premise not in usedPremises:
                # Pre-processing each premise in order to efficiently compare it against a given claim
                premiseTokens = Preprocessing(premise, "component")
                # Comparing how similar a given claim is to a premise (Calcuted via WMD)
                similarity = wordVectors.wmdistance(claimTokens, premiseTokens)
                # Adding each comparison to a list
                comparisons.append([str(claim), str(premise), similarity])
                # Used as a benchmark
                answer = comparisons[0]

        # Looping through the results for a give claim
        for item in comparisons:
            if item[2] < answer[2]:
                answer = item
        # Adding premise to used list
        usedPremises.append(answer[1])
        
        # Adding Components and their similarity to relations (list)
        relations.append([str(answer[0]), str(answer[1]), answer[2]])
    
    # Creating a new list to store premises that have not been used the first time round
    for premise in premises:
        if premise not in usedPremises:
            leftoverPremises.append(premise)
    
    # Attempt Two
    for leftoverPremise in leftoverPremises:
        # Check to see if it hasn't already been assigned (linked to a claim)
        if leftoverPremise not in usedPremises:
            # Pre-processing each premise in order to efficiently compare it against a given claim
            premiseTokens = Preprocessing(leftoverPremise, "component")
            # Stores comparisons between a given premise and claims
            comparisons = []
            for claim in claims:
                # Pre-processing each claim in order to efficiently compare it against a premise
                claimTokens = Preprocessing(claim, "component")
                # Comparing how similar a given claim is to a premise (Calcuted via WMD)
                similarity = wordVectors.wmdistance(claimTokens, premiseTokens)
                # Adding each comparison to a list
                comparisons.append([str(claim), str(leftoverPremise), similarity])
                # Used as a benchmark
                answer = comparisons[0]
            
        # Was having problems when we don't find any claims, quick solution
        if len(claims) != 0:
            # Looping through the results for a give claim
            for item in comparisons:
                if item[2] < answer[2]:
                    answer = item
            # Adding premise to used list
            usedPremises.append(answer[1])
            # Adding Components and their similarity to relations (list)
            relations.append([str(answer[0]), str(answer[1]), answer[2]])
    
    # Returning a list of Claims, supported by a given premise and their similartity score
    return relations

def SADFace(relations):
    # Testing SADFace Implementation via library
    sf.set_config_location("etc/canary.cfg")
    sf.sd = sf.init()

    sf.set_title("Canary")
    sf.add_notes("Canary x SADFace")
    sf.set_description("Canary findings outputted in SADFace")
    
    # Main loop taking information from Canary Relations
    for relation in relations:
        # We set the claim (conclusion) and premise (prem)
        con = str(relation[0])
        prem = [str(relation[1])]
        # We create an argument linking both of these components together
        arg = sf.add_support(con_text=con, prem_text=prem, con_id=None, prem_id=None)

    #print(sf.prettyprint())
    # Outputting changes to JSON file
    jsonData = sf.export_json()
    with open("output/canarySADFace.json", "w") as jsonFile:
        jsonFile.write(jsonData)
        print("JSON FILE WRITTEN")

def exportCSV(data, method):
    """ Exporting data from Canary to a .csv file for inspectation/graphing """
    # Creating .csv file
    if method == "canary":
        with open("canary/output/canary.csv", "a") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(data)
            # Closing file
            csvFile.close()
    elif method == "brat":
        with open("canary/output/brat.csv", "a") as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(data)
            # Closing file
            csvFile.close()
    else:
        print("WRONG USE OF FUNCTION.")
    
    
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
    directory = "data/corpus/"

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
    directory = "data/corpus/"

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

def bratTest(directory):
    """ Function used to extract Argumentative Components from the .ann file to find relations """

    # Stores the files that match those types (same filename has both .txt & .ann)
    files = []

    for file in os.listdir(directory):
        if file.endswith(".ann"):
            files.append(file)
  
    for file in files:
        # Printing file for testing
        print(file + "\n")
        
        # Getting Components from .ann file
        components = readAnn(directory + "/" + file)

        claims = components[1]
        premises = components[2]

        # Finding relations between these components
        relations = Relations(claims, premises)

        # Need to read in Ann Relations
        analysisRelations = readAnnRelations(directory + "/" + file)

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
        # Exporting results to .csv file
        data = []
        data.append([file, "Brat Relations", relationsCount, analysisRelationsCount])
        # Exporting to .csv
        exportCSV(data, "brat")
        print("File: " + file + " Relations Found: " + str(counts[0][0]) + "/" + str(counts[0][1]))

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
        # Used for Output
        f = (filename.split("/")[1])
        # Comparing Components results (Canary vs "Gold Standard")
        componentsAnalysis = BratAnalysis(filename + ".txt", filename + ".ann")
        # Comparing Relations results (Canary vs "Gold Standard")
        relationsAnalysis = BratRelationAnalysis(filename + ".txt", filename + ".ann")
        # Exporting results to .csv file
        data = []
        data.append([f, "Canary", str(componentsAnalysis[0][0]), str(componentsAnalysis[0][1]), str(componentsAnalysis[0][2]), str(relationsAnalysis[0][0])])
        data.append([f, "Manual", str(componentsAnalysis[1][0]), str(componentsAnalysis[1][1]), str(componentsAnalysis[1][2]), str(relationsAnalysis[0][1])])
        # Add another line for f1-score for components/relations
        exportCSV(data, "canary")
        print("File: " + filename + " exported to canaryTest.csv")
