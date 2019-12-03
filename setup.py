#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = []

setup_requirements = []

test_requirements = []

setup(
    author="Dileep Kishore, Muzamil Khan",
    author_email="dkishore@bu.edu",
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="A pytorch implementation of the DEEPred algorithm",
    # entry_points={"console_scripts": ["deepred_pytorch=deepred_pytorch.cli:main",],},
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="deepred_pytorch",
    name="deepred_pytorch",
    packages=find_packages(include=["deepred_pytorch", "deepred_pytorch.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/dileep-kishore/deepred_pytorch",
    version="0.1.0",
    zip_safe=False,
)
