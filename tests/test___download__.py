from unittest import TestCase
from canary.corpora import download_corpus
from urllib.error import HTTPError


class Test(TestCase):

    def test_download_corpus_exception(self):
        with self.assertRaises(Exception):
            download_corpus('corpus_that_does_not_exist')

    def test_download(self):
        try:
            download_corpus('us2016', overwrite_existing=True)
        except HTTPError:
            self.fail("Error downloading corpus")
