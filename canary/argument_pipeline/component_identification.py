from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC

from canary.argument_pipeline.model import Model
from canary.corpora import load_essay_corpus
from canary.preprocessing import Lemmatizer
from canary.preprocessing.transformers import DiscourseMatcher, \
    LengthTransformer, LengthOfSentenceTransformer, \
    AverageWordLengthTransformer, UniqueWordsTransformer, CountPosVectorizer, \
    WordSentimentCounter, EmbeddingTransformer, FirstPersonIndicatorMatcher


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

        model = Pipeline([
            ('feats', FeatureUnion([
                ('tfidvectorizer',
                 TfidfVectorizer(ngram_range=(1, 3), tokenizer=Lemmatizer(), lowercase=False)),
                ('length_10', LengthTransformer(12)),
                ('contain_claim', DiscourseMatcher('claim')),
                ('contain_premise', DiscourseMatcher('premise')),
                ('contain_major_claim', DiscourseMatcher('major_claim')),
                ('forward', DiscourseMatcher('forward')),
                ('thesis', DiscourseMatcher('thesis')),
                ('rebuttal', DiscourseMatcher('rebuttal')),
                ('backward', DiscourseMatcher('backward')),
                ('length_of_sentence', LengthOfSentenceTransformer()),
                ('average_word_length', AverageWordLengthTransformer()),
                ("fp", FirstPersonIndicatorMatcher()),
                ("eee", CountPosVectorizer()),
                ("eeeareae", UniqueWordsTransformer()),
                ("pos_sent", WordSentimentCounter("pos")),
                ("neg_sent", WordSentimentCounter("neg")),
                ("neu_sent", WordSentimentCounter("neu")),
                ("embeddings", EmbeddingTransformer())
            ])),
            ('stan', MaxAbsScaler()),
            ('clf', SVC(
                # gamma='auto',
                kernel='linear',
                random_state=0,
                probability=True,
            ))
        ])

        super(ArgumentComponent, self).train(pipeline_model=model,
                                             train_data=train_data,
                                             test_data=test_data,
                                             train_targets=train_targets,
                                             test_targets=test_targets,
                                             save_on_finish=True)
