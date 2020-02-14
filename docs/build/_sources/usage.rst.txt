Usage
=====

How to use/run ANA ?

Requirements
------------
Python dependencies:

- numpy
- pickle
- sklearn (https://scikit-learn.org/stable/)
- tpot (https://epistasislab.github.io/tpot/)
- gensim (https://radimrehurek.com/gensim/)
- pymongo (WIP)

Data
----
ANA requires textual and annotated data samples.
ANA considers dataset with the following structures:

- **texts** folder: contains raw text per identifier.
- **clauseClasses.csv**: contains the labels associated to clause granules.
- **sentClasses.csv**: contains the labels associated to sentence granules.
- **clauseProp.csv**: informative document containing titles and original 
  contract of clauses.

Each sample is associated to an identifier for each granularity.
The identifier consists of a set of descending integer (from higher granularity
to more specific ones) concatenated with underscores.
For example, if we consider granularities A>B>C, identifier are structured as:
*idA_idB_idC*.
In ANA experiments, we used clauses and sentences, and consequently used 
identifier such as: *0005_001*, refering to sentence 1 in clause 5.
Text files are named *identifier.txt*. The annotation csv files are tables
with the identifier in the first column, and the labels in the others.
Multiple labels can be considered at once, for each granularity.

.. todo::
    The annotations files should be automacally named according to the
    granularity level.

Such dataset can be provided manually, yet, the *annotation* folder in the 
project provides a tool to annotate texts (from pdf) and generates such 
documents. Note that this tool was used to preliminar annotation in early stages 
of ANA.
This tool load a directory of containing subfolder containing contracts and
remarks.

.. todo::
    Discuss the knowledge folder.


Training
--------
How to train each phase ?
The code for training (thus generating ensemble classifiers) can be find in the
package *training*. The module *GlobalTraining* is the one to be called.

::

    python -m ana\training\GlobalTraining.py

Running *GlobalTraining* calls three functions, one per phase:
*generatePhaseOnePipeline*, *generatePhaseTwoPipeline* and 
*generatePhaseThreePipeline* (1, 2 and 3 respectively correspond to 
:math:`\alpha`, :math:`\beta` and :math:`\gamma`).
Comment/uncomment the phases that are to be trained or not.
Generated models are exported in the model folder and used by the classification 
process.
Note that the dataset location is set in preparation modules.


Classification
--------------

Clause classificatrion is performed by the *ClauseAnalyzer* module in
package *evaluation*.
It uses the generated models to classify one clause.
An evaluation process uses a test set to compute the overall performance of ANA.

The evaluation process is executed when running *ClauseAnalyzer*::

    python -m ana\evaluation\ClauseAnalyzer.py

To analyse one clause, the function *analyzeClause* can be called::

    type, isAnom, anomType = analyzeClause(clause)

It provides the infered type of the clause, the abnormality of the clause, and
the type of anomaly (violated policies).


