# Imports
from nltk.tokenize import sent_tokenize, word_tokenize
import json
import sys
import os

from gensim import downloader as data
from nltk.corpus import stopwords
from nltk import download
download('stopwords')

def canaryLocal(file):
    """ Finds Argumentative Components in a local file """

    # Store Components
    claim = []
    majorClaim = []
    premise = []
    components = []

    # Importing Indicators via "indicators.json"
    with open("indicators.json") as indicatorsFile:
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
        
        # User file imported, converted to lowercase and case ignored (NEED TO ADD CASE IGNORE)
        userFile = open(file).read().lower()

        # Tokenizing userFile into sentences, also removes Whitespace/Breaks
        sentenceTokens = sent_tokenize(userFile)

        # Looping through userFile Tokens (sentences)
        for line in xrange(0, len(sentenceTokens)):
            # Claim Indicators loop
            for i in range(len(claimIndicators)):
                # Indicator found in a given sentence
                if claimIndicators[i] in sentenceTokens[line]:
                    # Store current Component
                    claimComponent = str(sentenceTokens[line])
                    # Check to see if Component is already in list
                    if claimComponent not in str(claim):
                        # Add to found claims
                        claim.append(claimComponent)

            # Major Indicators loop
            for i in range(len(majorClaimIndicators)):
                # Indicator found in a given sentence
                if majorClaimIndicators[i] in sentenceTokens[line]:
                    # Store current Component
                    claimMajorComponent = str(sentenceTokens[line])
                    # Check to see if Component is already in list
                    if claimMajorComponent not in str(majorClaim):
                        # Add to found claims
                        majorClaim.append(claimMajorComponent)

            # Premise Indicators loop
            for i in range(len(premiseIndicators)):
                # Indicator found in a given sentence
                if premiseIndicators[i] in sentenceTokens[line]:
                    # Store current Component
                    premiseComponent = str(sentenceTokens[line])
                    # Check to see if Component is already in list
                    if premiseComponent not in str(premise):
                        # Add to found claims
                        premise.append(premiseComponent)

    # All components add to a list to be returned/re-used in other functions
    components.append(claim)
    components.append(majorClaim)
    components.append(premise)
    return components

def canaryRelations(claims, premises):
    """ Finds Argumentative Relations from a list of Claims/Premises """

    # Store Relations
    relations = []

    # Inputting pre-trained data from Wikipedia 2014+ (word-vectors)
    wordVectors = data.load("glove-wiki-gigaword-100")

    # Setting up Stop words via NLTK
    stopWords = stopwords.words('english')

    # Import SADFace template (JSON File)
    with open("./canarySADFace.json") as jsonFile:
        SADFace = json.load(jsonFile)

    # Loop through all of the premises
    for premise in premises:
        # Tokenization (should switch to wordtokenizer by NLTK?)
        premiseTokens = premise.lower().split()
        # Removing Stop words
        premiseTokens = [w for w in premiseTokens if w not in stopWords]

        # Loop through claims in comparison to a given premise
        for claim in claims:
            # Tokenization (should switch to wordtokenizer by NLTK?)
            claimTokens = claim.lower().split()
            # Removing Stop words
            claimTokens = [w for w in claimTokens if w not in stopWords]
            # Comparing how similar a given claim is to a premise (Calcuted via WMD)
            similarity = wordVectors.wmdistance(claimTokens, premiseTokens)
            # Adding Components and their similarity to relations (list)
            relations.append([str(claim), str(premise), similarity])
    
    # Testing Prints
    for relation in relations:
        print("Relation: " + str(relation))
    

        
        

""" Used for testing the various functions """
if __name__ == "__main__":
    
    # Finding Components via Canary
    canary = canaryLocal(".././corpus/essay001.txt")  
    
    # Claim
    claims = canary[0]
    # Premise
    premises = canary[2]

    canaryRelations(claims, premises)