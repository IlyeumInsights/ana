"""Module PipelineTraining
*****************************

Train several pipelines according to multiple provided configurations.
That is to say that it trains a possible ensemble classifier.
.. todo:: Use MLflow to track the model trained.
"""

import os
import logging
import pickle
from joblib import dump
from joblib import load
import numpy as np


from sklearn.svm import LinearSVC, OneClassSVM, SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import BernoulliNB
# from sklearn.feature_selection import SelectPercentile, f_classif, RFE
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from tpot import TPOTClassifier

from ana.preparation import DataPreparation, DataSelection, DataCleaning
from ana.utils.Election import Election
from ana import Settings


def generateClassifReport(y_test, y_pred):
    """generateClassifReport

    Generate a report of the classifier performance based on its outputs and
    reference (test) labels.

    :param y_test: Expected labels.
    :type y_test: np.array
    :param y_pred: Predicted labels.
    :type y_pred: np.array
    :return: The human readable report.
    :rtype: str
    """
    classifReport = "\n-----\n"
    classifReport += 'CLASSIFICATION REPORT: \n'
    classifReport += 'Accuracy:%.2f%%' % (accuracy_score(y_test, y_pred)*100)+"\n"    
    classifReport += classification_report(y_test, y_pred)
    classifReport += "\n"
    classifReport += "Predicted: \n"
    classifReport += str(y_pred)+"\n"
    classifReport += "Expected: \n"
    classifReport += str(y_test)+"\n"

    return classifReport

def trainClassifier(config, path, classifierName, automl=False):
    """trainClassifier

    Trains a classifier for a single configuration.

    :param config: Classifier configuration
    :type config: list(any)
    :param path: Location of classifier export.
    :type path: str (dirPath)
    :param classifierName: Name of the classifier that will be used to name the
        exported file.
    :type classifierName: str
    :param automl: Assert if single model AutoML (TPOT) should be used, 
        defaults to False.
    :type automl: bool, optional
    :return: The trained model
    :rtype: sklearn.model
    """
    x_train, x_test, y_train, y_test, dataprepReport = DataPreparation.prepareDataIly(config[0], config[1], config[2], config[3], config[4], config[5], config[6], config[7], datasetFolder=Settings.TRAIN_DATASET_LOC, trasnformSerial=os.path.join(path, classifierName+"_transform.pickle"), mappingExport=os.path.join(path, classifierName+"_mapping.pickle"))


    if not automl:
        # mlmodel = SVC(C=25.0, tol=1.0, probability=True)
        mlmodel = LinearSVC(C=25.0, dual=False, loss="squared_hinge", penalty="l2", tol=1.0)
        mlmodel.fit(x_train, y_train) 
        dump(mlmodel, os.path.join(path, classifierName+".joblib"))
    else:
        mlmodel = TPOTClassifier(generations=5, population_size=50, cv=5, n_jobs=6, verbosity=2, scoring='f1_macro', config_dict='TPOT sparse')
        mlmodel.fit(x_train, y_train)        
        dump(mlmodel.fitted_pipeline_, os.path.join(path, classifierName+".joblib"))


    y_pred = mlmodel.predict(x_test)

    classifReport = generateClassifReport(y_test, y_pred)

    with open(os.path.join(path, classifierName+"_granularity.txt"), "w") as reportFile:
        reportFile.write(str(config[2]))

    with open(os.path.join(path, classifierName+"_report.txt"), "w") as reportFile:
        reportFile.write(dataprepReport)
        reportFile.write(classifReport)

    # logging.debug(dataprepReport)
    # logging.debug(classifReport)

    return mlmodel

def trainPipeline(configs, path, automl=True):
    """trainPipeline

    Generate multiple classifier as part of the pipeline as well as voting
    procedure.

    :param configs: Configurations of all subclassifiers
    :type configs: list(list(any))
    :param path: Destination for the export of models
    :type path: str (dirPath)
    :param automl: Assert if AutoML should be used, defaults to True
    :type automl: bool, optional
    """
    count = 0
    for config in configs:
        count += 1
        trainClassifier(config, path, "model"+str(count), automl)


def evaluatePipeline(path, nbconf, config, voteSent=None, voteClause=None, voteMode=None):
    """evaluatePipeline

    Evaluate the quality of classification of a pipeline by loading a part of
    the dataset.

    :param path: Directory containing the models of the pipeline.
    :type path: str (dirPath)
    :param nbconf: Number of configurations considered.
    :type nbconf: int
    :param config: Configurations, contraining information to load the dataset.
    :type config: list(any)
    :param voteSent: Default election element at sentence level,
        defaults to None
    :type voteSent: str, optional
    :param voteClause: Default election element at clause level,
        defaults to None
    :type voteClause: str, optional
    :param voteMode: Type of election performed on clause (to aggregate
        outputs of models), defaults to None
    :type voteMode: str, optional

    :return: Achieved f1_score
    :rtype: float
    :return: Expected values
    :rtype: np.array
    :return: Predicted values
    :rtype: np.array
    """    
    # Load clause
    # TODO use a dedicated function to gather clauses 
    X, Y = DataSelection.loadData([config[1], config[0]], filter=config[3], granularity=config[2], datasetFolder=Settings.TEST_DATASET_LOC)

    # Generate model name according to number of configs used
    modelNames = []
    for i in range(1,nbconf+1):
        modelNames.append("model"+str(i))

    predVal = [] # per clause
    expeVal = [] # Expected annotation per clause

    # Go through test
    for id in X:
        clause = X[id]
        # logging.debug("\n Evaluating clause \n "+ clause)
        resPerModel = []
        for modelName in modelNames:
            vectorizer = pickle.load(open(os.path.join(path, modelName+"_transform.pickle"),"rb"))
            mlmodel = load(os.path.join(path, modelName+".joblib"))
            granularity = open(os.path.join(path, modelName+"_granularity.txt")).read()
            mapping = pickle.load(open(os.path.join(path, modelName+"_mapping.pickle"),"rb"))
            
            clause = DataCleaning.clean(clause)

            if int(granularity) == 0:
                granuls = np.array([clause])
            else: # setence
                granuls = np.array(clause.split("."))

            granuls = list(filter(None, granuls))

            vectext = vectorizer.transform(granuls)
            y_pred = mlmodel.predict(vectext)
            y_pred = y_pred.tolist()

            # Vote per sentence            
            if int(granularity) == 1:
                # res = vote(y_pred, atLeastOne=voteSent)
                if voteSent is not None:
                    mode = "ato"
                else:
                    mode = "std"
                ele = Election(mode=mode, resultlvl="min", default=voteSent)
                res = ele.counting(y_pred)
            else:
                res = y_pred[0]

            resPerModel.append(res)
        # Voting per clause
        # voteRes = vote(resPerModel, atLeastOne=voteClause)
        if voteMode is None:
            if voteClause is not None:
                voteMode = "ato"
            else:
                voteMode = "std"
        ele = Election(mode=voteMode, resultlvl="min", default=voteClause)
        voteRes = ele.counting(resPerModel)
        # No value found (ato)
        if voteRes is None:
            voteRes = -1
        predVal.append(voteRes)
        if Y[id] in mapping:
            expeVal.append(mapping[Y[id]]) # mapping in range
        else:
            # TODO display error
            expeVal.append(1)

    predVal = np.array(predVal)
    expeVal = np.array(expeVal)

    return f1_score(expeVal, predVal, average="macro"), expeVal, predVal

def evaluatePipelineDebug(path, nbconf):
    """evaluatePipelineDebug

    Pipeline evaluation for ML conception and debugging.

    :param path: Directory containing the models of the pipeline.
    :type path: str (dirPath)
    :param nbconf: Number of configurations considered.
    :type nbconf: int
    :return: Achieved f1_score
    :rtype: float
    """    
    # Load clause
    # TODO use a dedicated function to gather clauses
    X, Y = DataSelection.loadData([0, 1], filter=[[[], [], []], [[], []]], granularity=0, datasetFolder=Settings.TEST_DATASET_LOC)
    
    # X = {"0001":"Le présent contrat est résiliable par l'une ou l'autre des parties, de plein droit : - en cas de liquidation de biens, de cessation de paiements ou de règlements judiciaires. - 10 jours après une mise en demeure  restée infructueuse en cas de manquement dans l'exécution des obligations réciproques et notamment en cas de manquement aux obligations. - en cas de force majeure. - Si le client final refuse pour quel raison que ce soit de travailler avec le consultant initialement affecté auprès de lui. - lorsque le marché principal est lui-même résilié sans qu'il y ait de faute de Client."}
    # Y = {"0001":"delai"}

    # Generate model name according to number of configs used
    modelNames = []
    for i in range(1,nbconf+1):
        modelNames.append("model"+str(i))


    predVal = [] # per clause
    expeVal = [] # Expected annotation per clause

    # Go through test
    for id in X:
        clause = X[id]
        logging.debug("\n Evaluating clause \n "+ clause)
        resPerModel = []
        for modelName in modelNames:
            logging.debug(modelName)
            vectorizer = pickle.load(open(os.path.join(path, modelName+"_transform.pickle"),"rb"))
            mlmodel = load(os.path.join(path, modelName+".joblib"))
            granularity = open(os.path.join(path, modelName+"_granularity.txt")).read()
            mapping = pickle.load(open(os.path.join(path, modelName+"_mapping.pickle"),"rb"))
            
            clause = DataCleaning.clean(clause)

            if int(granularity) == 0:
                granuls = np.array([clause])
            else: # setence
                granuls = np.array(clause.split("."))

            granuls = list(filter(None, granuls))

            vectext = vectorizer.transform(granuls)
            y_pred = mlmodel.predict(vectext)

            # Vote per sentence            
            if int(granularity) == 1:
                logging.debug("Voting for sentence "+str(y_pred))
                # res = vote(y_pred, atLeastOne=0)
                ele = Election(mode="ato", resultlvl="min", default=0)
                res = ele.counting(y_pred)
            else:
                logging.debug("Classif of clause "+str(y_pred))
                res = y_pred[0]
        
            logging.debug(Y[id])
            if Y[id] in mapping:
                logging.debug(mapping[Y[id]])
            else:
                logging.debug("Unknown mapping for "+str(Y[id]))
            logging.debug(res)
            resPerModel.append(res)
        # Voting per clause
        logging.debug("Final Judgement: ")
        logging.debug(resPerModel)
        # voteRes = vote(resPerModel, atLeastOne=None)
        ele = Election(mode="std", resultlvl="min")
        voteRes = ele.counting(resPerModel)
        predVal.append(voteRes)
        if Y[id] in mapping:
            expeVal.append(mapping[Y[id]]) # mapping in range
        else:
            # TODO display error
            expeVal.append(-1)
        logging.debug(voteRes)

    predVal = np.array(predVal)
    expeVal = np.array(expeVal)
    logging.debug("Predicted")
    logging.debug(predVal)
    logging.debug("Expected")
    logging.debug(expeVal)
    logging.debug(classification_report(expeVal, predVal))

    return f1_score(expeVal, predVal, average="macro")



if __name__ == "__main__":
    # 1- ycol, 2- ygranulaity, 3- data granularity (0 clause, 1 sentence)
    # 4- filter/selection, 5- binary, 6- classes restirction, 7- oversample, 8- transformer
    configs = []
    # configs.append([1, 1, 1, [ [ [], ["R7", "None"], ["resiliation"] ], [ [], [] ] ], False, ["R7"], False, "count"])
    # configs.append([1, 0, 1, [ [ [], [], ["resiliation"] ], [ [], [] ] ], True, None, False, "count"])
    # configs.append([1, 0, 0, [ [ [], [], ["non_concurrence"] ], [ [], [] ] ], False, None, False, "hash"])
    # configs.append([1, 0, 0, [ [ [], ["R7", "None"], [] ], [ [], [] ] ], False, ["R7"], False, "count"])

    configs.append([1, 0, 0, [ [ [], [], [] ], [ [], [] ] ], True, None, False, "tfidf"])

    trainPipeline(configs, Settings.MODEL_PIPELINES, automl=False)

    evaluatePipelineDebug(Settings.MODEL_PIPELINES, len(configs))
