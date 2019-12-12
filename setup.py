#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name='cocktail',
    version='1.0',
    description=(
        'A blind source separation package using non-negative matrix factorization '
        'and non-negative ICA.'
    ),
    author='Marc Romaní',
    author_email='marcromani.ub@gmail.com',
    package_dir={'': 'src'},
    packages=find_packages(),
    install_requires=['scikit-learn']
)
