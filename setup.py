#!/usr/bin/python
# -*- coding: UTF-8 -*-

from setuptools import setup

def readme():
	with open("README.md", "r") as fh:
    	return fh.read()

setup(
    name="jlt",
    version="0.1",
    author="Ben Fauber",
    author_email="Ben.Fauber@dell.com",
    description="Johnson-Lindenstrauss transforms, random projections, and randomized Hadamard transforms in python 3.x",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/dell/jlt",
    packages=['jlt'],
    classifiers=[
        "Programming Language :: Python :: 3.x",
        "License :: Apache 2.0",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'math',
        'numpy',
        'scipy.sparse',
        'fht'
      ],
    python_requires='>=3.4',
    include_package_data=True,
    zip_safe=False
)
