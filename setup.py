from setuptools import setup, find_packages

setup(
    name='canary',
    version='0.0.1',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        "requests",
        "scikit-learn",
        "nltk",
    ],
    include_package_data=True,
    zip_safe=False
)
