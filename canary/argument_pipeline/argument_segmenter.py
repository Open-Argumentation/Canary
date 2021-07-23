import spacy
from canary.argument_pipeline.model import Model
from canary.corpora import load_essay_corpus

nlp = spacy.load('en_core_web_lg')


class ArgumentSegmenter(Model):
    def __init__(self, model_id=None, model_storage_location=None, load=True):
        if model_id is None:
            self.model_id = "arg_segmenter"

        super().__init__(
            model_id=self.model_id,
            model_storage_location=model_storage_location,
            load=load
        )

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None):
        corpus = load_essay_corpus()
        for doc in corpus:
            doc = nlp(doc.text)
            for j in doc:
                pass
            # doc = nlp(doc.text)
            # document_tokens = []
            #
            # for sentence in doc:
            #     document_tokens.append(sentence)
            pass
        pass


def is_in_arg_component():
    pass

def is_start_of_arg_component():
    pass



def word2features(sent, i):
    word = sent[i]

    features = {
        'bias': 1.0,
        'word.lower()': str(word).lower(),
        'word[-3:]': str(word)[-3:],
        'word[-2:]': str(word)[-2:],
        'word.is_lower': word.is_lower,
        'word.istitle()': word.is_title,
        'word.isdigit()': word.is_digit,
        'postag': word.pos_,
    }

    if i > 0:
        word1 = sent[i - 1]
        postag1 = word1.pos_
        features.update({
            '-1:word.lower()': str(word1).lower(),
            '-1:word.istitle()': word1.is_title,
            '-1:word.isupper()': word1.is_lower,
            '-1:postag': postag1,
            '-1:postag[:2]': str(postag1)[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1]
        postag1 = word1.pos_
        features.update({
            '+1:word.lower()': str(word1).lower(),
            '+1:word.istitle()': word1.is_title,
            '+1:word.isupper()': word1.is_lower,
            '+1:postag': postag1,
            '+1:postag[:2]': str(postag1)[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    sent = nlp(sent)
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, postag, label in sent]


def sent2tokens(sent):
    return [token for token, postag, label in sent]
