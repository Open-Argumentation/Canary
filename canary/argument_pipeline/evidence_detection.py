from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from canary.argument_pipeline.model import Model
from canary.preprocessing import Lemmatizer
from canary.preprocessing.transformers import UniqueWordsTransformer, \
    CountPunctuationVectorizer, SentimentTransformer, DiscourseMatcher, WordSentimentCounter, \
    AverageWordLengthTransformer, CountPosVectorizer, LengthOfSentenceTransformer


class EvidenceDetection(Model):

    def __init__(self, model_id=None, model_storage_location=None, load=True):
        if model_id is None:
            self.model_id = "evidence_detection"
        super().__init__(model_id=self.model_id, model_storage_location=model_storage_location, load=load)

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None):
        from canary.corpora import load_imdb_debater_evidence_sentences
        _train_data, train_targets, _test_data, test_targets = load_imdb_debater_evidence_sentences()

        train_text_only = []
        for train in _train_data:
            train_text_only.append(train["text"])

        test_text_only = []
        for test in _test_data:
            test_text_only.append(test["text"])

        model = Pipeline([
            ('features', FeatureUnion([
                ('bow',
                 TfidfVectorizer(
                     ngram_range=(1, 4),
                     tokenizer=Lemmatizer(),
                 )
                 ),
                ('pos_tagger', CountPosVectorizer()),
                ("length", LengthOfSentenceTransformer()),
                ("support", DiscourseMatcher(component="support")),
                ("conflict", DiscourseMatcher(component="conflict")),
                ("punctuation", CountPunctuationVectorizer()),
                ("sentiment", SentimentTransformer()),
                ("average_word_length", AverageWordLengthTransformer()),
            ])),
            ("m", MaxAbsScaler(copy=False)),
            ('svc',
             LinearSVC(random_state=0, C=9e12, tol=1e-6)
             )
        ])

        super().train(model, train_text_only, test_text_only, train_targets, test_targets)

    def features(self, train, test):
        uwt = UniqueWordsTransformer()
        pos_wsc = WordSentimentCounter("pos")
        neg_wsc = WordSentimentCounter("neg")
        av_word_length = AverageWordLengthTransformer()
        l = LengthOfSentenceTransformer()

        def get_features(data):
            features = []

            for d in data:
                features.append({
                    "len": l.transform([d])[0][0],
                    "disc_maj": DiscourseMatcher("major_claim").transform([d])[0][0],
                    "uwt": uwt.transform([d])[0][0],
                    "average_word_length": av_word_length.transform([d])[0][0],
                })

            return features

        train_data = get_features(train)
        test_data = get_features(test)

        return train_data, test_data
