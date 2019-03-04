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

    for claim in claims:
        # Tokenization (should switch to wordtokenizer by NLTK?)
        claimTokens = claim.lower().split()
        # Removing Stop words
        claimTokens = [w for w in claimTokens if w not in stopWords]

        # Stores comparisons between a given premise and claims
        comparisons = []

        for premise in premises:
            # Tokenization (should switch to wordtokenizer by NLTK?)
            premiseTokens = premise.lower().split()
            # Removing Stop words
            premiseTokens = [w for w in premiseTokens if w not in stopWords]
            # Comparing how similar a given claim is to a premise (Calcuted via WMD)
            similarity = wordVectors.wmdistance(claimTokens, premiseTokens)

            # Adding each comparison to a list
            comparisons.append([str(claim), str(premise), similarity])

        # Used as a benchmark
        answer = comparisons[0]
        
        # Looping through the 
        for item in comparisons:
            if item[2] < answer[2] and item[2] != 0.0:
                answer = item

        # Adding Components and their similarity to relations (list)
        relations.append([str(answer[0]), str(answer[1]), answer[2]])

    # Returning a list of Claims, supported by a given premise and their similartity score
    return relations
    

""" Used for testing the various functions """
if __name__ == "__main__":
    
    # Finding Components via Canary
    canary = canaryLocal(".././corpus/essay001.txt")  
    
    # Claim
    canaryClaims = canary[0]
    # Premise
    canaryPremises = canary[2]

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
    
    # Used to store premises found (Used for comparison against other list to find what premises are left)
    foundPremises = []

    # Used for storing left over premises
    leftoverPremises = []
    
    for premise in premises:
        for relation in relations:
            if relation[1] == premise:
                #print ("Premise found: " + relation[1])
                foundPremises.append(relation[1])
    
    for premise in premises:
        if premise not in foundPremises:
            # Checking to see if it is already in the list
                if premise not in leftoverPremises:
                    leftoverPremises.append(premise)
        
    print("\n")
    for relation in relations:
        print("Claim: " + relation[0] + " supported by Premise: " + relation[1] + " Similarity: " + str(relation[2]))
        print("\n")

    for premise in leftoverPremises:
        print("Leftover Premise: " + str(premise))
        print("\n")
    