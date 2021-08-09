import pandas
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC

from canary.argument_pipeline.model import Model
from canary.corpora import load_essay_corpus
from canary.preprocessing import Lemmatizer
from canary.preprocessing.transformers import DiscourseMatcher, \
    AverageWordLengthTransformer, WordSentimentCounter, EmbeddingTransformer, TfidfPosVectorizer


class ArgumentComponent(Model):
    """
    Detects argumentative components from natural language
    """

    def __init__(self, model_id: str = None, model_storage_location=None, load: bool = True):
        """
        :param model_id: the ID of the model
        :param model_storage_location: where the model should be stored
        :param load: Whether to automatically load the model
        """

        if model_id is None:
            self.model_id = "argument_component"
        else:
            self.model_id = model_id

        super().__init__(
            model_id=self.model_id,
            model_storage_location=model_storage_location,
            load=load
        )

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=False, *args, **kwargs):
        train_data, test_data, train_targets, test_targets = load_essay_corpus(purpose="component_prediction")

        train_dict, test_dict = self.prepare_dictionary_features(train_data, test_data)

        train_data = pandas.DataFrame(train_data)
        test_data = pandas.DataFrame(test_data)

        feats = FeatureUnion([
            ('tfidvectorizer',
             TfidfVectorizer(ngram_range=(1, 3), tokenizer=Lemmatizer(), lowercase=False)),
            ('posv', TfidfPosVectorizer(ngrams=2)),
            ('contain_claim', DiscourseMatcher('claim')),
            ('contain_premise', DiscourseMatcher('premise')),
            # ('contain_major_claim', DiscourseMatcher('major_claim')),
            ('forward', DiscourseMatcher('forward')),
            ('thesis', DiscourseMatcher('thesis')),
            ('rebuttal', DiscourseMatcher('rebuttal')),
            ('backward', DiscourseMatcher('backward')),
            ('average_word_length', AverageWordLengthTransformer()),
            ("fp_i", DiscourseMatcher('first_person')),
            ("pos_sent", WordSentimentCounter("pos")),
            ("neg_sent", WordSentimentCounter("neg")),
            ("neu_sent", WordSentimentCounter("neu")),
            ("embeddings", EmbeddingTransformer())
        ])

        feats.fit(train_data.cover_sentence.tolist())
        train_feats = feats.transform(train_data.cover_sentence.tolist())
        test_feats = feats.transform(test_data.cover_sentence.tolist())

        dict_feats = DictVectorizer().fit(train_dict)
        train_dict_feats = dict_feats.transform(train_dict)
        test_dict_feats = dict_feats.transform(test_dict)

        train_feats = hstack([train_feats, train_dict_feats])
        test_feats = hstack([test_feats, test_dict_feats])

        scaler = MaxAbsScaler().fit(train_feats)
        train_feats = scaler.transform(train_feats)
        test_feats = scaler.transform(test_feats)

        model = SVC(probability=True, random_state=0, class_weight='balanced', kernel='linear', C=10, gamma='auto')
        model = SGDClassifier(
            class_weight='balanced',
            random_state=0,
            loss='modified_huber',
        )

        super(ArgumentComponent, self).train(pipeline_model=model,
                                             train_data=train_feats,
                                             test_data=test_feats,
                                             train_targets=train_targets,
                                             test_targets=test_targets,
                                             save_on_finish=True)

    def prepare_dictionary_features(self, train, test):

        def get_features(data):
            features = []

            for d in data:
                features.append({
                    'is_intro': d['is_intro'],
                    'is_conclusion': d['is_conclusion']
                })

            return features

        train_data = get_features(train)
        test_data = get_features(test)

        return train_data, test_data
