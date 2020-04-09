# Structure of the code directory:

* knowledge: Datasets and knowledge management, all code related to data/knowledge acquisition and knowledge before the actual training/evaluation process. Typically, it handles files and database acquisition and access.
* preparation: Transform the data and get it ready for the training phase.
* training*: Perform the models trainings based on transformed data.
* evaluation*: Evaluate a clause based on the previously trained models.
* utils: Support and transversal functionnalities.
* rest*: Flask-based REST API server.
* experimental: Code used to experiment and try stuff.
