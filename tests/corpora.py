from unittest import TestCase
from canary.corpora import download_corpus, load_essay_corpus


class CorporaTest(TestCase):

    def test_download_corpus_exception(self):
        with self.assertRaises(Exception):
            download_corpus('corpus_that_does_not_exist')

    def test_load_essay_corpus(self):
        corpus = load_essay_corpus()
        assert corpus is not None
