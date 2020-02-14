Project Structure
=================

The project is structured as following:

- **ana**: Source code of ANA experimental prototype.
- **docs**: Documentation, including sources.
- **knowledge**: Background and expert provided knowledge including rules and
  vocabularies.
- **annotation**: Tool used for labeling texts (from pdf) and create a formatted
  dataset.
- **models**: Default export folder for models.
- **mlruns**: Default output directory for mlFlow (wip)

Source Package
--------------
ANA's code is divded into the following packages:

- **knowledge**: Datasets and knowledge (policies, vocabularies...) management.
  It includes all code related to data acquisition and storage before any ML
  process.
- **preparation**: Data preprocessing (tranformation, cleaning,...) before
  the training or evaluation.
- **training** [#f1]_ : Ensemble classifiers generation based on transformed data for
  the three phases.
- **evaluation** [#f1]_ : Evaluation of a single clause using the generated Ensemble
  classifiers.
- **utils**: Support and transveral functions.
- **experimental**: Code used to experiments features, models and methods.

.. [#f1] Packages including runnable processes.