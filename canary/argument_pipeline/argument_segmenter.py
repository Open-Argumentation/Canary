import nltk
import sklearn_crfsuite
from sklearn_crfsuite import metrics

from canary import logger
from canary.argument_pipeline.model import Model
from canary.corpora import load_essay_corpus


class ArgumentSegmenter(Model):
    def __init__(self, model_id=None, model_storage_location=None, load=True):
        if model_id is None:
            self.model_id = "arg_segmenter"

        super().__init__(
            model_id=self.model_id,
            model_storage_location=model_storage_location,
            load=load
        )

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=True):
        # Need to get data into a usable shape

        logger.debug("Getting raw data")
        train_data, test_data, train_targets, test_targets = load_essay_corpus(purpose="sequence_labelling")

        logger.debug("Getting training features")
        train_data = [get_sentence_features(s) for s in train_data]

        logger.debug("Getting training labels")
        train_targets = [get_labels(s) for s in train_targets]

        logger.debug("Getting test features")
        test_data = [get_sentence_features(s) for s in test_data]

        logger.debug("Getting test labels")
        test_targets = [get_labels(s) for s in test_targets]

        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            all_possible_transitions=True
        )

        crf.fit(train_data, train_targets)

        labels = list(crf.classes_)
        y_pred = crf.predict(test_data)
        metrics.flat_f1_score(test_targets, y_pred,
                              average='weighted', labels=labels)
        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )

        logger.debug("\n\n" + metrics.flat_classification_report(
            test_targets, y_pred, labels=sorted_labels, digits=4
        ))

        self.metrics = metrics.flat_classification_report(
            test_targets, y_pred, labels=sorted_labels, digits=4, output_dict=True
        )

        self.model = crf

        self.save({
            "model_id": self.model_id,
            "model": self.model,
            "metrics": self.metrics
        })

    def predict(self, data, probability=False):
        if probability is True:
            logger.warn(
                f"{self.__class__.__name__} does not support probability predictions. This parameter is ignored.")

        data_type = type(data)

        if data_type is str:
            data = nltk.word_tokenize(data)
            data = [get_sentence_features(data)]

        if data_type is list:
            if all(type(item) is dict for item in data) is False:
                logger.error("The list passed in needs to only contain dictionary features")
                return

        return super().predict(data, probability=False)


def get_word_features(sent, i):
    word = sent[i]

    features = {
        'bias': 1.0,
        'word.lower()': str(word).lower(),
        'word[-3:]': str(word)[-3:],
        'word[-2:]': str(word)[-2:],
        'word.is_lower': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': nltk.pos_tag([word])[0][1]
    }

    if i > 0:
        word1 = sent[i - 1]
        postag1 = nltk.pos_tag([word1])[0][1]
        features.update({
            '-1:word.lower()': str(word1).lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': str(postag1)[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1]
        postag1 = nltk.pos_tag([word1])[0][1]
        features.update({
            '+1:word.lower()': str(word1).lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': str(postag1)[:2],
        })
    else:
        features['EOS'] = True

    return features


def get_sentence_features(sent):
    return [get_word_features(sent, i) for i in range(len(sent))]


def get_labels(sent):
    return [label for label in sent]
