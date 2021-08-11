import multiprocessing
from setuptools import setup, find_packages
from setuptools.command.test import test as setup_test
import os
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "ml_experiment_manager",
    version = "0.1.2",
    #packages = find_packages(),
    test_suite='nose2.collector.collector',
    # Dependencies on other packages:
    setup_requires   = ['pytest-runner'],
    install_requires = [
                        'pandas>=1.1.3',
                        'numpy>=1.19.1',           
                        'torch>=1.7.1',       
                        'matplotlib>=3.3.0',
                        'nose2>=0.9.2',     # For testing
                        ],

    tests_require    =[
                       'testfixtures>=6.14.1',
                       ],

    # metadata for upload to PyPI
    author = "Andreas Paepcke",
    author_email = "paepcke@cs.stanford.edu",
    description = "machine learning",
    long_description_content_type = "text/markdown",
    long_description = long_description,
    license = "BSD",
    keywords = "machine learning",
    url = "https://github.com/paepcke/ml_experiment_manager.git"
)

print("To run tests, type 'nose2'")

