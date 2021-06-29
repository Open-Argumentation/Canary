import spacy
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from canary.argument_pipeline.model import Model
from canary.corpora import load_essay_corpus
from canary.preprocessing import PosLemmatizer
from canary.preprocessing.transformers import TfidfPunctuationVectorizer, DiscourseMatcher, EmbeddingTransformer, \
    SentimentTransformer


class StructurePredictor(Model):

    def __init__(self, model_id=None, model_storage_location=None, load=True):
        if model_id is None:
            self.model_id = "structure_predictor"

        super().__init__(model_id=self.model_id, model_storage_location=model_storage_location, load=load)

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None):
        _train_data, _test_data, train_targets, test_targets = load_essay_corpus(purpose="relation_prediction")

        _train_text_only = []
        _train_arg1_text_only = []
        _train_arg2_text_only = []
        _test_arg1_text_only = []
        _test_arg2_text_only = []

        for entry in _train_data:
            _train_arg1_text_only.append(entry["arg1_text"])
            _train_arg2_text_only.append(entry["arg2_text"])

        _test_text_only = []
        for entry in _test_data:
            _test_arg1_text_only.append(entry["arg1_text"])
            _test_arg2_text_only.append(entry["arg2_text"])

        text_feats = FeatureUnion([
            ("Count", TfidfVectorizer(ngram_range=(1, 3), lowercase=False, tokenizer=PosLemmatizer())),
            ("punct", TfidfPunctuationVectorizer()),
            ("support", DiscourseMatcher("support")),
            ("thesis", DiscourseMatcher("thesis")),
            ("conflict", DiscourseMatcher("conflict")),
            ("rebuttal", DiscourseMatcher("rebuttal")),
            ("sentiment_pos", SentimentTransformer("pos")),
            ("sentiment_neg", SentimentTransformer("neg")),
            ("sentiment_neu", SentimentTransformer("neu")),
            ("vector_norm", EmbeddingTransformer()),
        ])

        text_feats.fit(_train_arg1_text_only)
        t1 = text_feats.transform(_train_arg1_text_only)
        t2 = text_feats.transform(_test_arg1_text_only)

        text_feats.fit(_train_arg2_text_only)
        t3 = text_feats.transform(_train_arg2_text_only)
        t4 = text_feats.transform(_test_arg2_text_only)

        _train_data_dict, _test_data_dict = self.prepare_dictionary_features(_train_data, _test_data)

        f = DictVectorizer().fit(_train_data_dict)
        f1 = f.transform(_train_data_dict)
        f2 = f.transform(_test_data_dict)

        combined_features_train = hstack([t1, t3, f1])
        combined_features_test = hstack([t2, t4, f2])

        scaler = StandardScaler(with_mean=False).fit(combined_features_train)
        combined_features_train = scaler.transform(combined_features_train)
        combined_features_test = scaler.transform(combined_features_test)

        model = SGDClassifier(
            random_state=0,
            early_stopping=True,
            n_jobs=2,
            alpha=0.000001
        )

        # model = AdaBoostClassifier(
        #     base_estimator=SGDClassifier(
        #         random_state=0,
        #         early_stopping=True,
        #         n_jobs=2,
        #         alpha=0.000001
        #     ),
        #     random_state=0,
        #     n_estimators=30,
        #     algorithm='SAMME'
        # )

        super(StructurePredictor, self).train(
            pipeline_model=model,
            train_data=combined_features_train,
            test_data=combined_features_test,
            train_targets=train_targets,
            test_targets=test_targets
        )

    def prepare_dictionary_features(self, train, test):
        nlp = spacy.load('en_core_web_lg')

        def get_features(data):
            features = []

            for d in data:
                sent1 = nlp(d["arg1_text"])
                sent2 = nlp(d["arg2_text"])
                features.append({
                    "arg1_start": d["arg1_start"],
                    "arg2_start": d["arg2_start"],
                    "arg1_end": d["arg1_end"],
                    "arg2_end": d["arg2_end"],
                    "arg1_type": d["arg1_type"],
                    "arg2_type": d["arg2_type"],

                    "sentence_similarity_norm": sent1.similarity(sent2),
                })

            return features

        train_data = get_features(train)
        test_data = get_features(test)

        return train_data, test_data
