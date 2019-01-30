# Imports
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import json
import sys
import os


# Need to split this up, have vars as global
def canaryLocal(file):

    # Counts used for Output (Only for console application)
    sentence_count = 0
    claim_major_count = 0
    claim_count = 0
    claim_for_count = 0
    claim_against_count = 0
    premise_count = 0

    # Used to store possible Argument Components
    possible_claim_major = []
    possible_claim = []
    possible_claim_for = []
    possible_claim_against = []
    possible_premise = []

    # Used to store Indicators via 'indicators.json'
    claim_major_indicators = []
    claim_indicators = []
    premise_indicators = []

    # Importing Indicators from JSON file
    with open('indicators.json') as f:
        data = json.load(f)
        # For loop to loop through JSON file to add the correct Indicators to their array
        for i in xrange(0, len(data["indicators"])):
            # Major Indicators
            for i in range(len(data["indicators"][i]["major"])):
                claim_major_indicators.append(
                    str(data["indicators"][0]["major"][i]))
            i = 0
        
            # Claim Indicators
            for i in range(len(data["indicators"][i]["claim"])):
                claim_indicators.append(str(data["indicators"][0]["claim"][i]))
            i = 0
            
            # Premise Indicators
            for i in range(len(data["indicators"][i]["premise"])):
                premise_indicators.append(str(data["indicators"][0]["premise"][i]))

    # Variable for user file, converts to lower case to match Indicators (need to ignore case)
    local_file = open('.././corpus/' + file).read().lower()
    # Tokenizing the Data into sentences (Removes Whitespace/Breaks)
    sentence_list = sent_tokenize(local_file)

    # Looping through each sentence from the file
    for line in xrange(0, len(sentence_list)):
        # Incrementing Sentence Count
        sentence_count += 1
        # Looping through all of the Indicators (Major)
        for i in range(len(claim_major_indicators)):
            # If one of the Indicators is in a given sentence
            if claim_major_indicators[i] in sentence_list[line]:
                # Increase the count of possible Major Claims
                claim_major_count += 1
                # Create a var for current sentence
                sentence = str(sentence_list[line])
                # Split the sentence at the Indicator and remove Whitespace
                claim_major = sentence.split(
                    claim_major_indicators[i], 1)[1].lstrip()
                # Add to Major Claims
                possible_claim_major.append(str("[" + claim_major + "]"))
        # Looping through all of the Indicators (Claim)
        for i in range(len(claim_indicators)):
            if claim_indicators[i] in sentence_list[line]:
                # Increase the count of possible Claims
                claim_count += 1
                sentence2 = str(sentence_list[line])
                # Split
                claim = sentence2.split(claim_indicators[i], 1)[1].lstrip()
                # Add to Claims
                possible_claim.append(str("[" + claim + "]"))
        # Looping through all of the Indicators (Premise)
        for i in range(len(premise_indicators)):
            if premise_indicators[i] in sentence_list[line]:
                # Increase the count of possible Premises
                premise_count += 1
                sentence3 = str(sentence_list[line])
                # Add to Premises
                possible_premise.append(str("[" + sentence3 + "]"))


    # Ouputs
    print ("Sentence Count: " + str(sentence_count))
    print ("Claim Count [Major]: " + str(claim_major_count))
    print ("Claim Count: " + str(claim_count))
    print ("Claim Count [For]: " + str(claim_for_count))
    print ("Claim Count [Against]: " + str(claim_against_count))
    print ("Premise Count: " + str(premise_count))

    # Name for output file
    #print("Testing File Name: " + file)
    # Spliting file at the '.' to extract the local file name
    
    """ WORKS FOR (1), but not (2)
    f = file.split('/') 
    filename = (f[3].split('.'))
    file_final = str("Arguments_" + filename[0] + ".txt")
    """
    f = str(file.split('/'))
    filename = (f.split('.')) 
    file_final2 = filename[0].split("'")
    file_final = str("Arguments_" + file_final2[1] + ".txt")

    directory = '.././canary/output/' + file_final
    # Writing possible Argument Components to a file
    output = open(directory, "w+")

    # Looping through all of the Arguments (adding them to the file 'Arguments.txt')
    for claims in possible_claim_major:
        output.write("Possible Claim [Major]: \r" + str(claims))
        output.write("\n")
        output.write("\n")
    for claims2 in possible_claim:
        output.write("Possible Claim: \r" + str(claims2))
        output.write("\n")
        output.write("\n")
    for premise in possible_premise:
        output.write("Possible Premise: \r" + str(premise))
        output.write("\n")
        output.write("\n")
    output.close()

    # Addressing the user as to where the potential arguments are stored
    print ("Possible Argument Components written to File: " + file_final)

    # Need to return Argument Component count to display against manual analysis (Function Attribute)
    canaryLocal.major = str(claim_major_count)
    canaryLocal.claim = str(claim_count)
    canaryLocal.premise = str(premise_count)

    