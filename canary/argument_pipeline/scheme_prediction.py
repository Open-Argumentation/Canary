from canary.argument_pipeline.model import Model
import canary.corpora


class SchemePredictior(Model):

    def __init__(self, model_id=None, model_storage_location=None, load=True):
        if model_id is None:
            self.model_id = "arg_scheme_predictor"
        super().__init__(model_id, model_storage_location, load=load)

    def train(self, pipeline_model=None, train_data=None, test_data=None, train_targets=None, test_targets=None,
              save_on_finish=False, **kwargs):

        corpus = canary.corpora.load_araucaria_corpus(purpose='scheme_prediction')

        super().train(pipeline_model, train_data, test_data, train_targets, test_targets, save_on_finish)
