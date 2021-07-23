import nltk
import pandas
from sklearn.linear_model import SGDClassifier

from canary import logger
from canary.argument_pipeline.model import Model
from canary.corpora import load_ukp_sentential_argument_detection_corpus
from canary.preprocessing.transformers import UniqueWordsTransformer, SentimentTransformer, DiscourseMatcher, \
    WordSentimentCounter, AverageWordLengthTransformer


class ForAgainst(Model):
    def __init__(self, model_id=None, model_storage_location=None, load=True):
        if model_id is None:
            self.model_id = "for_against"

        super().__init__(model_id=self.model_id, model_storage_location=model_storage_location, load=load)

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None):
        train_data, train_targets, test_data, test_targets = self.prepare_corpus()

        train_data = pandas.DataFrame(train_data)
        test_data = pandas.DataFrame(test_data)

        sgd = SGDClassifier(
            loss='log',
            random_state=0,
            warm_start=True,
            early_stopping=True,
        )
        model = sgd

        super(ForAgainst, self).train(pipeline_model=model,
                                      train_data=train_data,
                                      test_data=test_data,
                                      train_targets=train_targets,
                                      test_targets=test_targets)

    def prepare_corpus(self) -> tuple:
        test_data = []
        test_targets = []
        train_data = []
        train_targets = []

        logger.info("Preparing features")

        corpus = load_ukp_sentential_argument_detection_corpus(multiclass=True)
        uwt = UniqueWordsTransformer()
        sentiment_analyser = SentimentTransformer()
        conflictIndicator = DiscourseMatcher("conflict")
        supportIndicator = DiscourseMatcher("support")
        rebuttalIndicator = DiscourseMatcher("rebuttal")
        pos_wsc = WordSentimentCounter("pos")
        neg_wsc = WordSentimentCounter("neg")
        neu_wsc = WordSentimentCounter("neu")
        av_word_length = AverageWordLengthTransformer()

        for topic in corpus.keys():
            for k in corpus[topic]:
                if k == 'test':
                    for item in corpus[topic][k]:
                        test, target = item[0], item[1]
                        test_data.append({
                            "topic": topic,
                            "word_1": test.split()[0],
                            "word_1_pos": nltk.pos_tag(test.split())[0][1],
                            "last_word": test.split()[-1],
                            "last_word_pos": nltk.pos_tag(test.split())[-1][1],
                            "sen_length_g_5": len(test) > 5,
                            "average_word_length": av_word_length.transform([test])[0][0],
                            "uniqueWords": uwt.transform([test])[0][0],
                            "sentiment": sentiment_analyser.transform([test])[0][0],
                            "number_of_pos_words": pos_wsc.transform([test])[0][0],
                            "number_of_neg_words": neg_wsc.transform([test])[0][0],
                            "number_of_neu_words": neu_wsc.transform([test])[0][0],
                            "conflict_indicator": conflictIndicator.transform([test])[0][0],
                            "support_indicator": supportIndicator.transform([test])[0][0],
                            "rebuttal_indicator": rebuttalIndicator.transform([test])[0][0]
                        })
                        test_targets.append(target)
                if k == 'train':
                    for item in corpus[topic][k]:
                        train, target = item[0], item[1]
                        train_data.append(
                            {
                                "topic": topic,
                                "word_1": train.split()[0],
                                "word_1_pos": nltk.pos_tag(train.split())[0][1],
                                "last_word": train.split()[-1],
                                "last_word_pos": nltk.pos_tag(train.split())[-1][1],
                                "sen_length_g_5": len(train) > 5,
                                "average_word_length": av_word_length.transform([train])[0][0],
                                "uniqueWords": uwt.transform([train])[0][0],
                                "sentiment": sentiment_analyser.transform([train])[0][0],
                                "number_of_pos_words": pos_wsc.transform([train])[0][0],
                                "number_of_neg_words": neg_wsc.transform([train])[0][0],
                                "number_of_neu_words": neu_wsc.transform([train])[0][0],
                                "conflict_indicator": conflictIndicator.transform([train])[0][0],
                                "support_indicator": supportIndicator.transform([train])[0][0],
                                "rebuttal_indicator": rebuttalIndicator.transform([train])[0][0]
                            }
                        )
                        train_targets.append(target)

        return train_data, train_targets, test_data, test_targets
