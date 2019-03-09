import sys
import os

def test():
    indicators = ["hi", "te"]
    sentences = ["highly important", "This is a test", ]
    
    # Main Loop, Need to run though sentences and see if an indicator is "in" one of the sentences, if so print sentenance + indicator

    for sentence in sentences:
        
        for indicator in indicators:
            if indicator in sentence:
                print("Match:")
                print("Indicator: " + indicator + " Sentence: " + sentence)

def contains(string, word):
    return (" " + word + " ") in (" " + string + " ")


""" Used for testing the various functions """
if __name__ == "__main__":
    test = contains("highlight important", "important")
    print(str(test))