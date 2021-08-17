import pathlib

from setuptools import setup, find_packages

README = (pathlib.Path(__file__).parent / "README.md").read_text()

setup(
    name='canary-am',
    version='0.0.1',
    packages=find_packages(exclude=['tests', 'etc']),
    description="A Simple Argument Mining Library",
    long_description=README,
    python_requires=">=3.8",
    long_description_content_type="text/markdown",
    license="MIT",
    install_requires=[
        "requests",
        "scikit-learn",
        "nltk",
        "vaderSentiment",
        "pybrat",
        "spacy",
        "pandas",
        "benepar"
    ],
    url='http://openargumentation.org',
    project_urls={
        'Source': 'https://github.com/chriswales95/Canary',
    },
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
