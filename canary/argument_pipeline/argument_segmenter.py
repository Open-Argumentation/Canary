import nltk
import sklearn_crfsuite
from sklearn_crfsuite import metrics

import canary
import canary.utils
from canary.argument_pipeline.model import Model
from canary.corpora import load_essay_corpus
from canary.preprocessing.nlp import nltk_download


class ArgumentSegmenter(Model):

    def __init__(self, model_id=None, model_storage_location=None):
        if model_id is None:
            model_id = "arg_segmenter"

        super().__init__(
            model_id=model_id,
            model_storage_location=model_storage_location,
        )

    @staticmethod
    def default_train():
        # Need to get data into a usable shape

        canary.utils.logger.debug("Getting training data")
        train_data, test_data, train_targets, test_targets = load_essay_corpus(
            purpose="sequence_labelling",
            train_split_size=0.8
        )

        canary.utils.logger.debug("Getting training features")
        train_data = [get_sentence_features(s) for s in train_data]

        canary.utils.logger.debug("Getting training labels")
        train_targets = [get_labels(s) for s in train_targets]

        canary.utils.logger.debug("Getting test features")
        test_data = [get_sentence_features(s) for s in test_data]

        canary.utils.logger.debug("Getting test labels")
        test_targets = [get_labels(s) for s in test_targets]

        return train_data, test_data, train_targets, test_targets

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=True, *args, **kwargs):

        if any(item is None for item in [train_data, test_data, train_targets, test_targets]):
            # get default data if the above is not present
            train_data, test_data, train_targets, test_targets = self.default_train()

        canary.utils.logger.debug("Training algorithm")

        if pipeline_model is None:
            pipeline_model = sklearn_crfsuite.CRF(
                algorithm='pa',
                all_possible_transitions=True,
                all_possible_states=True
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

        canary.utils.logger.debug("\n\n" + metrics.flat_classification_report(
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
            canary.utils.logger.warn(
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
                canary.utils.logger.error("The list passed in needs to only contain dictionary features")
                return

        return super().predict(data, probability=False)

    def get_components_from_document(self, document: str) -> list:
        from nltk.tokenize.treebank import TreebankWordDetokenizer

        detokenizer = TreebankWordDetokenizer()

        # Segment from full text
        components = []
        current_component = []
        sentences = canary.corpora.essay_corpus.tokenize_essay_sentences(document)
        if len(sentences) < 2:
            canary.utils.logger.warn("There doesn't seem to be much to analyse in the document.")

        predictions = [self.predict(sentence) for sentence in sentences]

        # @TODO Ensure this works properly
        for prediction in predictions:
            for i, token in enumerate(prediction):
                if token[1] == "Arg-B":
                    current_component = [token[0]]
                elif token[1] == "Arg-I":
                    current_component.append(token[0])
                    if i < len(prediction):
                        if prediction[i + 1][1] == "O":
                            components.append(current_component)
                            current_component = []

        # Delete these
        del current_component
        del prediction
        del predictions

        canary.utils.logger.debug(f"{len(components)} components found from segmenter.")

        # Get covering sentences
        for i, component in enumerate(components):
            for sen in sentences:
                if detokenizer.detokenize(component) in sen or all(x in nltk.word_tokenize(sen) for x in component):
                    components[i] = {
                        'component_ref': i,
                        "cover_sentence": sen,
                        "component": detokenizer.detokenize(component),
                        "len_component": len(component),
                        "len_cover_sen": len(nltk.word_tokenize(sen)),
                        "tokens": component
                    }
                    split = components[i]['cover_sentence'].split((components[i]['component']))
                    try:
                        components[i].update({
                            'n_following_comp_tokens': len(nltk.word_tokenize(split[0])),
                            'n_preceding_comp_tokens': len(nltk.word_tokenize(split[1])),
                        })
                    except IndexError:
                        # kind of hackish but detokenising isn't a perfect process
                        cs = detokenizer.detokenize(nltk.word_tokenize(components[i]['cover_sentence']))
                        split = cs.split((components[i]['component']))
                        try:
                            components[i].update({
                                'n_following_comp_tokens': len(nltk.word_tokenize(split[0])),
                                'n_preceding_comp_tokens': len(nltk.word_tokenize(split[1])),
                            })
                        except IndexError as e:
                            # @TODO Fix this bit
                            canary.utils.logger.error(e)
                            components[i].update({
                                'n_following_comp_tokens': 0,
                                'n_preceding_comp_tokens': 0,
                            })

        paragraphs = [p.strip() for p in document.split("\n") if p and not p.isspace()]
        canary.utils.logger.debug(f"{len(paragraphs)} paragraphs in document.")

        if not all('tokens' in c for c in components):
            raise KeyError("There was an error finding argumentative components")

        for i, component in enumerate(components):
            for j, para in enumerate(paragraphs):
                if detokenizer.detokenize(component['tokens']) in para or all(
                        x in nltk.word_tokenize(para) for x in component['tokens']):
                    components[i].update({
                        'len_paragraph': len(nltk.word_tokenize(para)),
                        'para_ref': j + 1,
                        'is_in_intro': True if j == (0 or 1) else False,
                        'is_in_conclusion': True if j == (len(paragraphs) - 1) else False,
                    })

        # find n_following_components and n_preceding_components
        for component in components:
            if 'para_ref' not in component:
                raise ValueError("failed to find ...")
            neighbouring_components = [c for c in components if
                                       c['para_ref'] == component['para_ref'] and c != component]
            if len(neighbouring_components) < 2:
                component['n_following_components'] = 0
                component['n_preceding_components'] = 0
                component['component_position'] = 1
                component['first_in_paragraph'] = True
                component['last_in_paragraph'] = True

            else:
                component['n_preceding_components'] = len(
                    [c for c in neighbouring_components if c['component_ref'] < component['component_ref']])
                component['n_following_components'] = len(
                    [c for c in neighbouring_components if c['component_ref'] > component['component_ref']])
                component['component_position'] = (len(neighbouring_components) - component[
                    'n_following_components']) + 1
                component['first_in_paragraph'] = True if component["n_preceding_components"] == 0 else False
                component['last_in_paragraph'] = True if component["n_following_components"] == 0 else False

        for c in components:
            del c['tokens']
        return components


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

    if i > 2:
        word3 = sent[i - 3][0]
        postag3 = sent[i - 3][1]
        ent3 = sent[i - 3][2]
        features.update({
            '-3:word.lower()': str(word3).lower(),
            '-3:word.istitle()': word3.istitle(),
            '-3:word.isupper()': word3.isupper(),
            '-3:postag': postag3,
            '-3:postag[:2]': str(postag3)[:2],
            '-3:ent': ent3,
        })

    if i < len(sent) - 3:
        word3 = sent[i + 3][0]
        postag3 = sent[i + 3][1]
        ent3 = sent[i + 3][2]
        features.update({
            '+3:word.lower()': str(word3).lower(),
            '+3:word.istitle()': word3.istitle(),
            '+3:word.isupper()': word3.isupper(),
            '+3:postag': postag3,
            '+3:postag[:2]': str(postag3)[:2],
            '+3:ent': ent3,
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
    canary.preprocessing.nlp.nltk_download(['averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])
    from nltk.chunk import tree2conlltags
    return tree2conlltags(nltk.ne_chunk(nltk.pos_tag(sen)))
