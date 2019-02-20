# Test file to deal with the various techniques we can use to attempt to match Argumentative Components together

"""

Working Implementation of WMD with Hardcoded Components
Achieves 5/6 Correct relations in essay001

"""

# Imports
from gensim import downloader as data
from nltk.corpus import stopwords
from nltk import download
download('stopwords')

# Inputting pre-trained data from Wikipedia 2014+ (word-vectors)
word_vectors = data.load("glove-wiki-gigaword-100")

# Setting up Stop words via NLTK
stop_words = stopwords.words('english')

# Hard-coded "Gold Standard" components from 'essay001'
Claims = ["through cooperation, children can learn about interpersonal skills which are significant in the future life of all students", "competition makes the society more effective",
          "without the cooperation, there would be no victory of competition"]

Premises = ["What we acquired from team work is not only how to achieve the same goal with others but more importantly, how to get along with others",
            "During the process of cooperation, children can learn about how to listen to opinions of others, how to communicate with others, how to think comprehensively, and even how to compromise with other team members when conflicts occurred",
            "All of these skills help them to get on well with other people and will benefit them for the whole life",
            "the significance of competition is that how to become more excellence to gain the victory",
            "when we consider about the question that how to win the game, we always find that we need the cooperation",
            "Take Olympic games which is a form of competition for instance, it is hard to imagine how an athlete could win the game without the training of his or her coach, and the help of other professional staffs such as the people who take care of his diet, and those who are in charge of the medical care"]


print("-------- CANARY ARGUMENT RELATIONS --------")

count = 0
for claim in Claims:
        count += 1
        print("Claim " + str(count) + ": " + claim)


count = 0
for premise in Premises:
        count += 1
        print("Premise " + str(count) + ": " + premise)


premise_count = 0
for premise in Premises:
        premise_count += 1
        count = 0
        premise = premise.lower().split()
        # Removing Stop words
        premise = [w for w in premise if w not in stop_words]
        list = []
        for claim in Claims:
                count += 1
                claim = claim.lower().split()
                # Removing Stop words
                claim = [w for w in claim if w not in stop_words]

                similarity = word_vectors.wmdistance(claim, premise)
                list.append([premise_count, count,  similarity])
        #print(list)
        test = list[0]
        
        for item in list:
                if item[2] < test[2]:
                        test = item
        
        print ("Premise " + str(test[0]) + " supports Claim: " + str(test[1]) + " (Similarity: " + str(test[2]) + ")")
        
     


