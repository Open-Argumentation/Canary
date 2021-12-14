import pathlib

from setuptools import setup, find_packages

README = (pathlib.Path(__file__).parent / "README.md").read_text()

setup(
    packages=find_packages(exclude=['tests', 'etc']),
    long_description=README,
    long_description_content_type="text/markdown"
)
