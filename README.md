![ANA](ana_logo.png?raw=true)

# AutoML-based Negociation Assistant 

[![Python](https://img.shields.io/badge/Python-3.7.1-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

ANA is an assitance tool for contract-based negotiation processes. Based on annotated examples of rejected/accepted clauses (usually obtainable through tracks of previous negotiations) and acceptance policies, ANA learns to identify unacceptable clauses and to determine which policies are violated in the contract. In order to cope with the heterogenity of contracts, ANA relies on a composite AutoML approach.

ANA is presented in the paper: [Toward a Generic AutoML-Based Assistant for Contracts Negotiation, to appear in ECAI 2020 proceedings](https://ecai2020.eu/papers/738_paper.pdf).

# Documentation
[![made-with-sphinx-doc](https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg)](https://www.sphinx-doc.org/)

The documentation can be found in the "docs\build".

# Dependencies
### Major:
* scikit-learn
* gensim

### Utilities:
* unidecode
* csv
* numpy
* pickle
