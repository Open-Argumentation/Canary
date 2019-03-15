# Imports
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim import downloader as data
import json
import sys
import os
import random

def canaryPreprocessing(file, type):
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
        with open("stopwords.json") as stopWordsFile:
            sWords = json.load(stopWordsFile)

             # Used to store Stopwords
            stopWords = []

            # Looping through JSON file to add to list
            for i in xrange(0, len(sWords["stopwords"])):
                stopWords.append(str(sWords["stopwords"][i]))

        """
        # Setting up Stop words via NLTK
        stopWords = stopwords.words('english')
        """

        # Tokenization (should switch to wordtokenizer by NLTK?)
        tokens = file.lower().split()
        # Removing Stop words
        tokens = [w for w in tokens if w not in stopWords]
        return tokens
    else:
        print("SEE DOCUMENTATION FOR CORRECT USAGES")


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
        
        # Importing and pre-processing the User's input file
        sentenceTokens = canaryPreprocessing(file, "text")

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

def canaryRelations(claims, premises):
    """ Finds Argumentative Relations from a list of Claims/Premises """

    # Works better than the last one, but every claim must be linked to a claim!
    # But  I can't loop by claim if I have more premises

    # Store Relations
    relations = []
    # Stores used premises
    usedPremises = []
    
    # Inputting pre-trained data from Wikipedia 2014+ (word-vectors)
    wordVectors = data.load("glove-wiki-gigaword-100")

    # Attempt Two
    for premise in premises:
        # Check to see if it hasn't already been assigned (linked to a claim)
        if premise not in usedPremises:
            # Pre-processing each premise in order to efficiently compare it against a given claim
            premiseTokens = canaryPreprocessing(premise, "component")
            # Stores comparisons between a given premise and claims
            comparisons = []
            for claim in claims:
                # Pre-processing each claim in order to efficiently compare it against a premise
                claimTokens = canaryPreprocessing(claim, "component")
                # Comparing how similar a given claim is to a premise (Calcuted via WMD)
                similarity = wordVectors.wmdistance(claimTokens, premiseTokens)
                # Adding each comparison to a list
                comparisons.append([str(claim), str(premise), similarity])
            # Used as a benchmark
            answer = comparisons[0]

            # Looping through the results for a give claim
        for item in comparisons:
            if item[2] < answer[2] and item[2] != 0.0:
                answer = item
        # Adding premise to used list
        usedPremises.append(answer[1])

        # Adding Components and their similarity to relations (list)
        relations.append([str(answer[0]), str(answer[1]), answer[2]])

    # Returning a list of Claims, supported by a given premise and their similartity score
    return relations

def canarySupports():
    """ Able to find what Claims support or oppose the given topic """
    print("CANARY SUPPORTS")

def canarySADFace(relations):
    """ Function used to output found Components and Relations in SADFace format """
    
    # Reading in JSON SADFace Template
    with open('./canarySADFace.json') as jsonFile:
        SADFace = json.load(jsonFile)

    # Need to loop through canaryRelations output and find out what premises are linked to what claims
    
    for relation in relations:
        # Randomly generate id's for each component
        id = random.randint(1, 1000)
        claimId = random.randint(1, 1000)
        premiseId = random.randint(1, 1000)

        # Creating a node for claim
        SADFace['nodes'].append({
            "id": str(claimId), 
            "metadata": {}, 
            "sources": [], 
            "text": str(relation[0]), 
            "type": "atom"
        })
        
        # Creating a node for premise
        SADFace['nodes'].append({
            "id": str(premiseId), 
            "metadata": {}, 
            "sources": [], 
            "text": str(relation[1]), 
            "type": "atom"
        })
        # Need to link the above elements
        SADFace['edges'].append({
            "id": str(id), 
            "source_id": str(premiseId), 
            "target_id": str(claimId)
        })

    # Outputting changes to JSON file
    with open('./canarySADFace.json', 'w') as f:
        json.dump(SADFace, f, indent=4)
    



if __name__ == "__main__":
    """ Used for testing the various functions """

    # Finding Components via Canary
    canary = canaryLocal(".././corpus/essay001.txt")  
    
    # Claim
    canaryClaims = canary[1]
    # Premise
    canaryPremises = canary[2]
    # Major
    canaryMajor = canary[0]

    # Hard-coded "Gold Standard" components from 'essay001'
    claims = ["through cooperation, children can learn about interpersonal skills which are significant in the future life of all students", "competition makes the society more effective",
          "without the cooperation, there would be no victory of competition"]

    premises = ["What we acquired from team work is not only how to achieve the same goal with others but more importantly, how to get along with others",
            "During the process of cooperation, children can learn about how to listen to opinions of others, how to communicate with others, how to think comprehensively, and even how to compromise with other team members when conflicts occurred",
            "All of these skills help them to get on well with other people and will benefit them for the whole life",
            "the significance of competition is that how to become more excellence to gain the victory",
            "when we consider about the question that how to win the game, we always find that we need the cooperation",
            "Take Olympic games which is a form of competition for instance, it is hard to imagine how an athlete could win the game without the training of his or her coach, and the help of other professional staffs such as the people who take care of his diet, and those who are in charge of the medical care"]

    # Finding Relations between Components
    relations = canaryRelations(claims, premises)

    # Test print
    for relation in relations:
        print("Relation:" + "\n")
        print("Claim: " + str(relation[0]))
        print("Premise: " + str(relation[1]))
        print("Similarity: " + str(relation[2]))
        print("\n")
    
    