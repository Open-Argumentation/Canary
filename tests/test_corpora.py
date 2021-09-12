from unittest import TestCase
from canary.corpora import download_corpus, load_essay_corpus


class CorporaTest(TestCase):
    """
    Tests for canary.corpora
    """

    def test_download_corpus_exception(self):
        """
        Test that errors are raised if we try to download a corpus we don't know about
        """

        with self.assertRaises(Exception):
            download_corpus('corpus_that_does_not_exist')

    def test_load_essay_corpus(self):
        """
            Test loading the essay corpus. Should return something and not be None.
        """

        corpus = load_essay_corpus()
        assert corpus is not None
