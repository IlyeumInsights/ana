"""Module ClauseAnalyzer
************************

Performs clause classification, from phase 1 (type) to phase 3 (anomaly type).
All models shall be already trained before using this module.

.. todo:: In the future, it should include a server that receives a clause,
    classifies it and sends back the result of classification.

:Functions:
    analyzeClause:
        Analyze a given clause and provides its infered type,
        anomaly and violated policy

:Author: Nathan Ramoly
"""
import os
import pickle
import logging
from joblib import load
import numpy as np

from sklearn.metrics import classification_report

from ana.preparation import DataSelection, DataCleaning
from ana.utils.Election import Election
from ana import Settings


def evaluateModels(clause, phaseDir, modelNames, sentVote):
    """evaluateModels

    Applies a clause on a set of models, of a given phase, that perform
    similar classifications.
    Clauses can be analyze according to various granularity.
    When model works on a smaller scale (than the clause), results are
    aggregated back at clause level through an election process.

    :param clause: The textual clause. Mandatory argument.
    :type clause: str
    :param phaseDir: Path to the directory containing the models of the
        selected evaluation phase.
    :type phaseDir: str (filepath)
    :param modelNames: List of models to apply to the clause.
    :type modelNames: list(str)
    :param modelNames: Sentence election mode, currently not used.
    :type modelNames: str

    :return:
        - The list of classification outcomes per model (list(int)).
        - The list of mappings label/class (list(map)).
    """
    resPerModel = []
    mappings = []
    for modelName in modelNames:
        vectorizer = pickle.load(open(os.path.join(phaseDir, modelName+"_transform.pickle"), "rb"))
        mlmodel = load(os.path.join(phaseDir, modelName+".joblib"))
        granularity = open(os.path.join(phaseDir, modelName+"_granularity.txt")).read()
        mappings.append(pickle.load(open(os.path.join(phaseDir, modelName+"_mapping.pickle"), "rb")))

        clause = DataCleaning.clean(clause)

        if int(granularity) == 0:
            granuls = np.array([clause])
        else: # sentence
            granuls = np.array(clause.split("."))

        granuls = list(filter(None, granuls))

        vectext = vectorizer.transform(granuls)
        y_pred = mlmodel.predict(vectext)
        # try:
        #     print(mlmodel.classes_)
        #     print(mlmodel.predict_proba(vectext))
        # except AttributeError as err:
        #     print("AttributeError error: {0}".format(err))

        # Vote per sentence
        if int(granularity) == 1:
            # res = vote(y_pred, atLeastOne=None)
            vote = Election(mode="std", resultlvl="min")
            res = vote.counting(y_pred)
        else:
            res = y_pred[0]

        resPerModel.append(res)

    return resPerModel, mappings

def evaluateClauseTypeVocab(clause, mapping):
    """evaluateClauseTypeVocab

    Evaluates the clause by key word search rather than ML classification.
    Vocabulary is provided, for now, in a file at a fixed location.
    It is mainly used to infer the clause type.

    :param clause: The textual clause. Mandatory argument.
    :type clause: str
    :param mapping: Mapping between labels and class.
    :type mapping: dict(int,str)

    :return: Inferred class.
    :rtype: str
    """

    # vocabPath = f"Vocab\\"
    vocabPath = Settings.KL_VOCAB

    vocab = {}

    # Load vocab files/dict
    for clType in mapping:
        vocab[clType] = []
        # Check if file exist
        clTypeVocPath = os.path.join(vocabPath, clType+".txt")
        if os.path.isfile(clTypeVocPath):
            # Access file that is composed of words separated my spaces
            with open(clTypeVocPath, "r") as clTypeVocFile:
                vocab[clType] = clTypeVocFile.read().split()

    # Checking the clause
    # Go through each dict and assert if clause contains the key words
    detType = [] # determined type
    for clType in mapping:
        for kword in vocab[clType]:
            if kword in clause.split():
                # Adding if not already found
                if clType not in detType:
                    detType.append(clType)

    # Convert to id according to mapping
    # TODO, for now, only first one
    #if len(detType) > 0:
    if detType: # Does not work with numpy array
        return mapping[detType[0]]
    else:
        return None

def evaluatePhaseOne(clause, clauseTitle=None):
    """evaluatePhaseOne

    Performs the phase one of the classification.
    Uses trained models of phase one and key word research to infer the type
    of the clause.

    :param clause: The textual clause.
    :type clause: str
    :param clauseTitle: The title of the clause, that is used for key word
        search, defaults to 'None'.
    :type clauseTitle: str

    :return: The inferred type of the clause.
    :rtype: str
    """
    logging.info("Evaluation Phase 1")

    phaseOneDir = os.path.join(Settings.MODEL_PIPELINES, f"phase1\\")
    # phaseOneDir = f"Models\\pipeline\\phase1\\"
    nbFilePerModel = 5

    nbFile = len(os.listdir(phaseOneDir))

    nbModel = int(nbFile / nbFilePerModel)

    modelNames = []
    for i in range(1,nbModel+1):
        modelNames.append("model"+str(i))

    resPerModel, mappings = evaluateModels(clause, phaseOneDir, modelNames, None)

    # logging.debug(clause)

    if clauseTitle is not None:
        # TODO merge matching (see following todo)
        resVoc = evaluateClauseTypeVocab(clauseTitle, mappings[0]) 
        # logging.debug(resVoc)
        if resVoc is not None:
            resPerModel.insert(0, resVoc)
        # logging.debug(resPerModel)

    vote = Election(mode="std", priority="first", resultlvl="min")
    voteRes = vote.counting(resPerModel)
    # voteRes = vote(resPerModel, atLeastOne=None)

    resClass = None
    # Only applies to phase one of course
    if voteRes != -1:
        # TODO check if mappings are matching
        for mapping in mappings:
            for key in mapping:
                if mapping[key] == voteRes:
                    resClass = key
    logging.debug("Votes type: "+str(resClass))

    return resClass

def evaluatePhaseTwo(clause, clauseType):
    """evaluatePhaseTwo

    Performs the phase two of the classification knowing the type of the clause.
    It uses trained models of phase two to assert if the clause is anomalous.

    :param clause: The textual clause.
    :type clause: str
    :param clauseType: Type of the clause inferred in phase one or provided.
    :type clauseType: str

    :return: True if the clause is detected as anomalous, False otherwise.
    :rtype: bool
    """
    logging.info("Evaluation Phase 2")

    if clauseType is None or clauseType == "None":
        clauseType = "default"

    phaseTwoDir = os.path.join(Settings.MODEL_PIPELINES, "phase2\\")
    phaseTwoDir = os.path.join(phaseTwoDir, clauseType)
    # phaseTwoDir = "Models\\pipeline\\phase2\\"+clauseType
    nbFilePerModel = 5

    nbFile = len(os.listdir(phaseTwoDir))
    nbModel = int(nbFile / nbFilePerModel)
    modelNames = []
    for i in range(1, nbModel+1):
        modelNames.append("model"+str(i))

    resPerModel, __ = evaluateModels(clause, phaseTwoDir, modelNames, 1)

    vote = Election(mode="std", priority="first", resultlvl="min")
    voteRes = vote.counting(resPerModel)
    # voteRes = vote(resPerModel, atLeastOne=None)

    # logging.debug(mappings)

    # if voteRes == 1:
    #     return True
    # else:
    #     return False
    return voteRes == 1

def evaluatePhaseThree(clause, clauseType):
    """evaluatePhaseThree

    Performs the phase three of the classification process knowing the type
    of the clause.
    It uses trained models of phase three to infer the policies violated by the
    clause.

    :param clause: The textual clause.
    :type clause: :py:class:`str`
    :param clauseType: Type of the clause inferred in phase one or provided.
    :type clauseType: :py:class:`str`

    :return: List of inferred violted policies
    :rtype: list(str)
    """
    logging.info("Evaluation Phase 3")

    if clauseType is None or clauseType == "None":
        clauseType = "default"

    phaseThreeDir = os.path.join(Settings.MODEL_PIPELINES, "phase3\\")
    phaseThreeDir = os.path.join(phaseThreeDir, clauseType)
    # phaseThreeDir = "Models\\pipeline\\phase3\\"+clauseType
    nbFilePerModel = 5

    nbFile = len(os.listdir(phaseThreeDir))
    nbModel = int(nbFile / nbFilePerModel)

    if nbModel < 0:
        with open(os.path.join(phaseThreeDir, "defclass.txt"), "r") as defClassFile:
            resClass = defClassFile.read()
        return resClass
    else:
        modelNames = []
        for i in range(1,nbModel+1):
            modelNames.append("model"+str(i))

        resPerModel, mappings = evaluateModels(clause, phaseThreeDir, modelNames, None)

        # TODO solve respermodel resulting in empty list
        vote = Election(mode="not", priority="first", resultlvl="ful", default=0)
        voteRes = vote.counting(resPerModel) # including all classes
        # voteRes = vote(resPerModel, atLeastOne=None)

        if voteRes is None or len(voteRes) <= 0:
            voteRes = [(-1, 0)]

        resClass = []
        # Only applies to phase one of course
        for v in voteRes:
            if v[0] != -1:
                # TODO check if mappings are matching
                tk = None
                for mapping in mappings:
                    for key in mapping:
                        if mapping[key] == v[0]:
                            tk = key
                resClass.append(tk)
        return resClass

def analyzeClause(clause, clauseTitle=None):
    """analyzeClause

    Core function of clause analysis.
    Performs the full analysis (phase 1 to 3) of a clause.
    It first determines the type of the clause (phase 1).
    Then according to the outcome of phase 1, determines if the clause is
    anomalous or correct.
    If the clause is detected an anomalous, phase 3 is performed to infer
    the violated policies, thus explaining the rejection of the clause.

    :param clause: The textual clause.
    :type clause: str
    :param clauseTitle: The title of the clause, that is used for key word
        search, defaults to 'None'.
    :type clauseTitle: str

    :return:
        - The inferred type of the clause (str).
        - True if the clause is detected as anomalous, False otherwise (bool).
        - List of inferred violted policies (list(str)).
    """
    logging.info("--- Entering evaluation process of clause: ---")
    logging.info(clause)

    clause = DataCleaning.clean(clause)

    clauseType = evaluatePhaseOne(clause, clauseTitle=clauseTitle)

    logging.info("Inferred clause type: "+str(clauseType))

    # if clauseType is None:
    #     return False, None

    isAnom = evaluatePhaseTwo(clause, clauseType)

    anomType = None

    if isAnom:
        logging.info("Clause is anomalous, infering violated policy...")
        anomType = evaluatePhaseThree(clause, clauseType)
        logging.info("Violated policy: "+str(anomType))
    else:
        logging.info("Clause is valid.")

    logging.info("--- End of clause evaluation process ---")
    return clauseType, isAnom, anomType


def analyzeClauseAlpha(clause, clauseTitle=None):
    """analyzeClauseAlpha

    Alpha function of clause analysis.
    Performs the phase 1 of analysis (phase 1 to 3) of a clause.
    It determines the type of the clause (phase 1).

    :param clause: The textual clause.
    :type clause: str
    :param clauseTitle: The title of the clause, that is used for key word
        search, defaults to 'None'.
    :type clauseTitle: str

    :return: The inferred type of the clause (str)
    """
    logging.info("--- Entering alpha evaluation process of clause: ---")
    logging.info(clause)

    clause = DataCleaning.clean(clause)

    clauseType = evaluatePhaseOne(clause, clauseTitle=clauseTitle)

    logging.info("Inferred clause type: "+str(clauseType))

    return clauseType


def analyzeClauseBeta(clause, clauseType):
    """analyzeClauseBeta

    Beata function of clause analysis.
    Performs the phase 2 of the analysis of the clause.
    According to the clause type, determines if the clause is
    anomalous or correct.

    :param clause: The textual clause.
    :type clause: str
    :param clausetype: The type of the clause, infered from phase 1 or given.
    :type clauseTitle: str

    :return: True if the clause is detected as anomalous, False otherwise (bool).

    """

    logging.info("--- Entering beta evaluation process of clause: ---")
    logging.info(clause)

    clause = DataCleaning.clean(clause)

    isAnom = evaluatePhaseTwo(clause, clauseType)

    return isAnom


def analyzeClauseGamma(clause, clauseType):
    """analyzeClauseGamma

    Gamma function of clause analysis.
    Performs the phase 3 of analysis of a clause.
    The clause is supposed anomalous, this function infers
    the violated policies, thus explaining the rejection of the clause.

    :param clause: The textual clause.
    :type clause: str
    :param clausetype: The type of the clause, infered from phase 1 or given.
    :type clauseTitle: str

    :return: List of inferred violted policies (list(str)).
    """

    logging.info("--- Entering beta evaluation process of clause: ---")
    logging.info(clause)

    clause = DataCleaning.clean(clause)

    anomTypes = evaluatePhaseThree(clause, clauseType)

    return anomTypes

def analyzeClauseBetaGamma(clause, clauseType):
    """analyzeClauseBetaGamma

    Beta and gamma function of clause analysis.
    Performs the phase 2 and 3 of analysis of a clause.
    According to the clause type, determines if the clause is
    anomalous or correct.
    If the clause is detected an anomalous, phase 3 is performed to infer
    the violated policies, thus explaining the rejection of the clause.


    :param clause: The textual clause.
    :type clause: str
    :param clausetype: The type of the clause, infered from phase 1 or given.
    :type clauseTitle: str

    :return:
        - True if the clause is detected as anomalous, False otherwise (bool).
        - List of inferred violted policies (list(str)).
    """

    logging.info("--- Entering beta & gamma evaluation process of clause: ---")
    logging.info(clause)

    clause = DataCleaning.clean(clause)

    isAnom = evaluatePhaseTwo(clause, clauseType)

    anomTypes = None

    if isAnom:
        anomTypes = evaluatePhaseThree(clause, clauseType)

    return isAnom, anomTypes

def convertLabelToVal(label, classList):
    """convertLabelToVal

    Convert a textual label to its numeric counterpart .

    :param label: Label to convert.
    :type label: str
    :param classList: List of classes.
    :type classList: list(str)

    :return: The numerical identifier of the label (that is its position in the
        list)
    :rtype: int
    """
    for i in range(0,len(classList)):
        if label == classList[i]:
            return i
    return -1

def evaluationOnTestSet():
    """evaluationOnTestSet

    Evaluates the quality of the process.
    Used for debugging and ML conception.
    """
    logging.info("Test process started...")

    X, Y = DataSelection.loadData([0, 1], filter=[ [ [], [], [] ], [ [], [] ] ], granularity=0, datasetFolder=Settings.TEST_DATASET_LOC)
    __, Yc = DataSelection.loadData([0, 2], filter=[ [ [], [], [] ], [ [], [] ] ], granularity=0, datasetFolder=Settings.TEST_DATASET_LOC)

    predictedType = []
    expectedType = []
    predictedEval = []
    expectedEval = []
    predictedEvalType = []
    expectedEvalType = []
    handleAnom = ["R2", "R3", "R4", "R5", "R7", "R8", "R9", "R11", "R11", "R12", "R15"]
    # handleAnom = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15", "R16", "R17", "R18", "R19", "R20"]
    clTypes = ["paiement", "resiliation", "non_sollicitation", "non_concurrence"] # TODO use rule loader

    for id in Y:
        # print(Y[id])
        if Y[id] in handleAnom:
            expectedEval.append(1)
        else:
            expectedEval.append(0)

        expectedEvalType.append(convertLabelToVal(Y[id], handleAnom))

    for id in Yc:
        # print(Yc[id])
        expectedType.append(convertLabelToVal(Yc[id], clTypes))


    for id in X:
        clause = X[id]
        # print("\n Evaluating clause \n "+ clause)

        clauseType, clauseEval, anomType = analyzeClause(clause)

        # print(anomType)

        predictedType.append(convertLabelToVal(clauseType, clTypes))

        # TODO non binary
        # No anom
        #  if clauseEval == False:
        if not clauseEval:
            predictedEval.append(0)
        # Anom
        else:
            predictedEval.append(1)

        vanomtype = []
        if anomType is not None:
            for anomt in anomType:
                vanomtype.append(convertLabelToVal(anomt, handleAnom))
        else:
            vanomtype.append(-1)
        predictedEvalType.append(vanomtype)

    # pre compare with expected value and keep best values
    # TODO with first value only
    predictedEvalTypeBest = [-1] * len(expectedEvalType)
    for i in range(0, len(expectedEvalType)):
        if expectedEvalType[i] in predictedEvalType[i]:
            predictedEvalTypeBest[i] = expectedEvalType[i]
        else:
            predictedEvalTypeBest[i] = predictedEvalType[i][0]

    predictedEvalTypeFirst = [-1] * len(predictedEvalType)
    for i in range(0, len(predictedEvalType)):
        predictedEvalTypeFirst[i] = predictedEvalType[i][0]

    predTyp = np.array(predictedType)
    expeTyp = np.array(expectedType)
    print("Predicted")
    print(predTyp)
    print("Expected")
    print(expeTyp)
    print(classification_report(expeTyp, predTyp))

    predVal = np.array(predictedEval)
    expeVal = np.array(expectedEval)
    print("Predicted")
    print(predVal)
    print("Expected")
    print(expeVal)
    print(classification_report(expeVal, predVal))

    predValT = np.array(predictedEvalTypeBest)
    predValTF = np.array(predictedEvalTypeFirst)
    expeValT = np.array(expectedEvalType)
    print("Predicted (first)")
    print(predValTF)
    print("Predicted (best)")
    print(predValT)
    print("Expected")
    print(expeValT)
    print("First")
    print(classification_report(expeValT, predValTF))
    print("Best")
    print(classification_report(expeValT, predValT))


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-8s'
                        '[%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.DEBUG)
    evaluationOnTestSet()
