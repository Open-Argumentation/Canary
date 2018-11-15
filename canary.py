from nltk.tokenize import word_tokenize

def Tokenizer(string):
    tokens = word_tokenize(string)
    print("Tokenizing Input...")
    return str(tokens)
