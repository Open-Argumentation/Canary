import nltk
import sklearn_crfsuite
from sklearn_crfsuite import metrics

import canary
from canary import logger
from canary.argument_pipeline.model import Model
from canary.corpora import load_essay_corpus
from canary.utils import nltk_download


class ArgumentSegmenter(Model):

    def __init__(self, model_id=None, model_storage_location=None, load=True):
        if model_id is None:
            model_id = "arg_segmenter"

        super().__init__(
            model_id=model_id,
            model_storage_location=model_storage_location,
        )

    @staticmethod
    def default_train():
        # Need to get data into a usable shape

        logger.debug("Getting training data")
        train_data, test_data, train_targets, test_targets = load_essay_corpus(purpose="sequence_labelling",
                                                                               train_split_size=0.7)

        logger.debug("Getting training features")
        train_data = [get_sentence_features(s) for s in train_data]

        logger.debug("Getting training labels")
        train_targets = [get_labels(s) for s in train_targets]

        logger.debug("Getting test features")
        test_data = [get_sentence_features(s) for s in test_data]

        logger.debug("Getting test labels")
        test_targets = [get_labels(s) for s in test_targets]

        return train_data, test_data, train_targets, test_targets

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=True, *args, **kwargs):

        if any(item is None for item in [train_data, test_data, train_targets, test_targets]):
            # get default data if the above is not present
            train_data, test_data, train_targets, test_targets = self.default_train()

        logger.debug("Training algorithm")

        if pipeline_model is None:
            pipeline_model = sklearn_crfsuite.CRF(
                algorithm='lbfgs',
                all_possible_transitions=True,
            )

        pipeline_model.fit(train_data, train_targets)

        labels = list(pipeline_model.classes_)
        y_pred = pipeline_model.predict(test_data)
        metrics.flat_f1_score(test_targets,
                              y_pred,
                              average='weighted',
                              labels=labels)

        sorted_labels = sorted(
            labels,
            key=lambda name: (name[1:], name[0])
        )

        logger.debug("\n\n" + metrics.flat_classification_report(
            test_targets, y_pred, labels=sorted_labels, digits=4
        ))

        self._metrics = metrics.flat_classification_report(
            test_targets, y_pred, labels=sorted_labels, digits=4, output_dict=True
        )

        self._model = pipeline_model
        if save_on_finish is True:
            self.save()

    def predict(self, data, probability=False, binary=False):
        """

        :param data:
        :param probability:
        :param binary: binary detection of arguments
        :return:
        """

        nltk_download(['punkt', 'averaged_perceptron_tagger'])
        if probability is True:
            logger.warn(
                f"{self.__class__.__name__} does not support probability predictions. This parameter is ignored.")

        data_type = type(data)

        if data_type is str:
            tokens = nltk.word_tokenize(data)
            data = [get_sentence_features(tokens)]
            predictions = super().predict(data, probability=False)[0]
            if binary is not None:
                if binary is True:
                    if all(k == "O" for k in super().predict(data, probability=False)[0]):
                        return False
                    else:
                        return True
            return list(zip(tokens, predictions))

        if data_type is list:
            if all(type(item) is dict for item in data) is False:
                logger.error("The list passed in needs to only contain dictionary features")
                return

        return super().predict(data, probability=False)


def get_word_features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': str(word).lower(),
        'word[-3:]': str(word)[-3:],
        'word[-2:]': str(word)[-2:],
        'word.is_lower': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': sent[i][1],
        'ent': sent[i][2]
    }

    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        ent1 = sent[i - 1][2]
        features.update({
            '-1:word.lower()': str(word1).lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': str(postag1)[:2],
            '-1:ent': ent1
        })
    else:
        features['BOS'] = True

    if i > 1:
        word2 = sent[i - 2][0]
        postag2 = sent[i - 2][1]
        ent2 = sent[i - 2][2]
        features.update({
            '-2:word.lower()': str(word2).lower(),
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            '-2:postag': postag2,
            '-2:postag[:2]': str(postag2)[:2],
            '-2:ent': ent2,
        })

    if i < len(sent) - 2:
        word2 = sent[i + 2][0]
        postag2 = sent[i + 2][1]
        ent2 = sent[i + 2][2]
        features.update({
            '+2:word.lower()': str(word2).lower(),
            '+2:word.istitle()': word2.istitle(),
            '+2:word.isupper()': word2.isupper(),
            '+2:postag': postag2,
            '+2:postag[:2]': str(postag2)[:2],
            '+2:ent': ent2,
        })

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        ent1 = sent[i + 1][2]
        features.update({
            '+1:word.lower()': str(word1).lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': str(postag1)[:2],
            '+1:ent': ent1,
        })
    else:
        features['EOS'] = True

    return features


def get_sentence_features(sent):
    sent = chunk(sent)
    return [get_word_features(sent, i) for i in range(len(sent))]


def get_labels(sent):
    return [label for label in sent]


def chunk(sen):
    canary.utils.nltk_download(['averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])
    from nltk.chunk import tree2conlltags
    return tree2conlltags(nltk.ne_chunk(nltk.pos_tag(sen)))
