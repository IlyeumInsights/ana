"""Module GlobalTraining
*****************************

Generate all pipelines for all three phases and all anomomalies
through an AutoML like approach directed by rules (politics).
Three phases:
- Classification of clause per (relevant) type
- Prediction of anomaly
- Classification of detected anomaly
It uses the policies as logic rules to orient the learning.

Functions
    generatePhaseOnePipeline
    generatePhaseTwoPipeline
    generatePhaseThreePipeline
"""
import os
import logging

import ana.training.PipelineTraining as pt
from ana.knowledge.RulesManager import RulesManager
from ana import Settings


def generatePhaseOnePipeline():
    """generatePhaseOnePipeline

    Determine and train the best pipeline for classifiying clauses according
    to their types.
    It explores several data preparation parameters (feature extraction,
    oversamplings) and granulairty.
    This pipeline always consider two subclassifiers, each dedicated to a
    granularity (sentence and clause).
    The pipeline is exported in the model directory.
    """
    logging.info("Generating phase 1 pipeline...")
    rm = RulesManager()

    classes = rm.gatherClass()
    logging.debug(classes)

    # Configuring according to: oversampling, and transformation
    transformers = ["count", "tfidf", "hash"]
    oversamplings = [True, False]

    f1ref = 0
    selConfig = []

    modelsPath = os.path.join(Settings.MODEL_PIPELINES, "phase1\\")

    for transformer1 in transformers:
        for transformer2 in transformers:
            for oversampling1 in oversamplings:
                for oversampling2 in oversamplings:
                    logging.info("Generating pipeline with: "+transformer1+" "+transformer2+" "+str(oversampling1)+" "+str(oversampling2))
                    # 1- ycol, 2- ygranulaity,
                    # 3- data granularity (0 clause, 1 sentence)
                    # 4- filter/selection, 5- binary, 6- classes restirction,
                    # 7- oversample, 8- transformer
                    configs = []
                    configs.append([2, 0, 0, [[[], [], []], [[], []]], False, classes, oversampling1, transformer1])
                    configs.append([2, 0, 1, [[[], [], []], [[], []]], False, classes, oversampling2, transformer2])
             

                    pt.trainPipeline(configs, modelsPath, automl=Settings.AUTOML_TPOT)
                    # Second part of config in not used
                    evalConf = [2, 0, 0, [[[], [], []], [[], []]], None, None, None, None]
                    # TODO try several vote
                    f1, ev, pv = pt.evaluatePipeline(modelsPath, len(configs), evalConf)

                    logging.info("Resulting f1: "+str(f1))

                    if f1 > f1ref:
                        f1ref = f1
                        selConfig = []
                        selConfig.extend(configs)

    # Select best config
    logging.debug(f1ref)
    logging.debug(selConfig)
    pt.trainPipeline(selConfig, modelsPath, automl=Settings.AUTOML_TPOT)
    evalConf = [2, 0, 0, [[[], [], []], [[], []]], None, None, None, None]
    res, ev, pv = pt.evaluatePipeline(modelsPath, len(selConfig), evalConf)
    logging.debug(res)
    logging.info(pt.generateClassifReport(ev, pv))


def generatePhaseTwoPipeline():
    """generatePhaseTwoPipeline

    Determine and train the best pipeline for detecting anomalies in clauses
    (phase 2).
    It explores several data preparation parameters (feature extraction,
    oversamplings) and granulairty.
    This pipeline always consider several subclassifiers combinations.
    The pipeline is exported in the model directory.
    """
    logging.info("Generating phase 2 pipeline...")
    rm = RulesManager()

    classes = rm.gatherClass()

    # Generate a pipeline for each classe:
    for clauseClass in classes:
        transformers = ["count", "tfidf", "hash"]
        oversamplings = [False]
        nbPipelineType = 3
        f1ref = 0

        modelsPath = os.path.join(Settings.MODEL_PIPELINES, "phase2\\")
        pipeDir = os.path.join(modelsPath, clauseClass)

        if not os.path.exists(pipeDir):
            os.makedirs(pipeDir)

        # Try multiple combinations of:
        # 1- clause of type clauseClass -> anom
        # 2- phrase of clause of type clausclass -> anom
        # 3- set of classif for each anom type (with sentence if rules compatible)
        # 4- All clauses and all anom

        for count in range(0, 2**nbPipelineType):
            for transformer in transformers:
                for oversampling in oversamplings:
                    logging.info("Generating pipeline with: "+transformer+" "+str(oversampling)+" "+str(count))

                    configs = []

                    # Default classifier on the whole clause:
                    if True:
                        configs.append([1, 0, 0, [[[], [], [clauseClass]], [[], []]], True, None, oversampling, transformer])
                        pass

                    # Classifier on phrase
                    if count&1 == 1:
                        configs.append([1, 1, 1, [[[], [], [clauseClass]], [[], []]], True, None, oversampling, transformer])
                        pass

                    # Set of binary classifiers
                    if count&2 == 2:
                        polids = rm.getPoliticsOfClass(clauseClass)
                        # TODO in some way, intermediate vote
                        for polid in polids:
                            configs.append([1, 0, 0, [ [ [], [polid, "None"], [clauseClass] ], [ [], [] ] ], False, [polid], False, transformer])
                            # configs.append([1, 0, 0, [ [ [], [], [clauseClass] ], [ [], [] ] ], False, [polid], False, transformer])
                            # TODO, leave sentence optionnal
                            # if rm.politicAppliesToSentence(polid):
                            #     configs.append([1, 1, 1, [ [ [], [polid, "None"], [clauseClass] ], [ [], [] ] ], False, [polid], False, transformer])

                        pass

                    # Classifier on all clauses
                    if count&4 == 4:
                        configs.append([1, 0, 0, [ [ [], [], [] ], [ [], [] ] ], True, None, oversampling, transformer])
                        pass

                    pt.trainPipeline(configs, pipeDir, automl=Settings.AUTOML_TPOT)
                    # Second part of config in not used
                    evalConf = [1, 0, 0, [ [ [], [], [clauseClass] ], [ [], [] ] ], None, None, None, None]
                    # TODO try several vote
                    votes = [None, 1]
                    for vote in votes:
                        f1, ev, pv = pt.evaluatePipeline(pipeDir, len(configs), evalConf, voteSent=1, voteClause=vote)

                        logging.info("Resulting f1: "+str(f1)+" with vote "+str(vote))

                        if f1 > f1ref:
                            f1ref = f1
                            selConfig = []
                            selvote = vote
                            selConfig.extend(configs)


        # Clear directory
        for f in os.listdir(pipeDir):
            os.remove(os.path.join(pipeDir, f))

        # Select best config
        logging.debug("\n"+clauseClass)
        logging.debug(f1ref)
        logging.debug(selConfig)
        pt.trainPipeline(selConfig, pipeDir, automl=Settings.AUTOML_TPOT)
        evalConf = [1, 0, 0, [ [ [], [], [clauseClass] ], [ [], [] ] ], None, None, None, None]
        res, ev, pv = pt.evaluatePipeline(pipeDir, len(selConfig), evalConf, voteSent=1, voteClause=selvote)
        logging.debug(res)
        logging.info(pt.generateClassifReport(ev, pv))

    # Generate default pipeline
    # TODO Add sentence
    pipeDir = os.path.join(modelsPath, "default")
    if not os.path.exists(pipeDir):
        os.makedirs(pipeDir)
    transformers = ["count", "tfidf", "hash"]
    oversamplings = [False]
    f1ref = 0
    selConfig = []

    for transformer in transformers:
        for oversampling in oversamplings:
            configs = []
            configs.append([1, 0, 0, [ [ [], [], [] ], [ [], [] ] ], True, None, oversampling, transformer])

            pt.trainPipeline(configs, pipeDir, automl=Settings.AUTOML_TPOT)
            # Second part of config in not used
            evalConf = [1, 0, 0, [ [ [], [], [] ], [ [], [] ] ], None, None, None, None]
            # TODO try several vote
            f1, ev, pv = pt.evaluatePipeline(pipeDir, len(configs), evalConf, voteSent=1, voteClause=None)

            logging.info("Default: Resulting f1: "+str(f1)+" with vote "+str(vote))

            if f1 > f1ref:
                f1ref = f1
                selConfig = []
                selvote = vote
                selConfig.extend(configs)     

    # Clear directory
    for f in os.listdir(pipeDir):
        os.remove(os.path.join(pipeDir, f))

    # Select best config
    logging.debug("\n"+"Default")
    logging.debug(f1ref)
    logging.debug(selConfig)
    pt.trainPipeline(selConfig, pipeDir, automl=Settings.AUTOML_TPOT)
    evalConf = [1, 0, 0, [[[], [], []], [[], []]], None, None, None, None]
    res, ev, pv = pt.evaluatePipeline(pipeDir, len(selConfig), evalConf, voteSent=1, voteClause=None)
    logging.debug(res)
    logging.info(pt.generateClassifReport(ev, pv))




def generatePhaseThreePipeline():
    """generatePhaseThreePipeline

    Determine and train the best pipeline for classifying anomalies of clauses
    (phase 3). In other words, it aims to determine the violated policies of
    an anomalous clause.
    It explores several data preparation parameters (feature extraction,
    oversamplings) and granulairty.
    This pipeline always consider several subclassifiers combinations.
    The pipeline is exported in the model directory.
    """
    logging.info("Generating phase 3 pipeline...")
    rm = RulesManager()

    classes = rm.gatherClass()

    # Generate a pipeline for each classe:
    for clauseClass in classes:
        transformers = ["count", "tfidf", "hash"]
        oversamplings = [False]
        nbPipelineType = 3
        f1ref = 0

        # Anom of class
        anoms = rm.getPoliticsOfClass(clauseClass)
        logging.debug(anoms)

        modelsPath = os.path.join(Settings.MODEL_PIPELINES, "phase3\\")
        pipeDir = os.path.join(modelsPath, clauseClass)
        # pipeDir = "Models\\pipeline\\phase3\\"+clauseClass

        if not os.path.exists(pipeDir):
            os.makedirs(pipeDir)

        if len(anoms) < 2:
            logging.warning("Ignoring pipe for anomaly "+str(anoms))
            # Creating file with default class
            with open(os.path.join(pipeDir, "defclass.txt"), "w") as defClassFile:
                if len(anoms) > 0:
                    defClass = anoms[0]
                else:
                    defClass = None
                defClassFile.write(defClass)
            continue

        # Try multiple combinations of:
        # 1- clause of type clauseClass -> anom
        # 2- phrase of clause of type clausclass -> anom
        # 3- set of classif for each anom type (with sentence if rules compatible)
        # 4- All clauses and all anom

        for count in range(0, 2**nbPipelineType):
            for transformer in transformers:
                for oversampling in oversamplings:
                    logging.info("Generating pipeline with: "+transformer+" "+str(oversampling)+" "+str(count))

                    configs = []

                    # Default classifier on the whole clause:
                    if True:
                        configs.append([1, 0, 0, [[[], anoms, [clauseClass]], [[], []]], False, None, oversampling, transformer])

                    # Classifier on phrase
                    if count&1 == 1:
                        configs.append([1, 1, 1, [[[], anoms, [clauseClass]], [[], []]], False, None, oversampling, transformer])

                    # Set of binary classifiers
                    if count&2 == 2:
                        polids = rm.getPoliticsOfClass(clauseClass)
                        # TODO in some way, intermediate vote
                        for polid in polids:
                            configs.append([1, 0, 0, [[[], [polid, "None"], [clauseClass]], [[], []] ], False, None, False, transformer])
                            # configs.append([1, 0, 0, [ [ [], anoms, [clauseClass] ], [ [], [] ] ], False, [polid], False, transformer])
                            # TODO, leave sentence optionnal
                            if rm.politicAppliesToSentence(polid):
                                configs.append([1, 1, 1, [[[], [polid, "None"], [clauseClass]], [[], [polid, "None"]]], False, None, False, transformer])

                    # Classifier on all clauses
                    if count&4 == 4:
                        configs.append([1, 0, 0, [[[], anoms, []], [[], []]], False, None, oversampling, transformer])

                    pt.trainPipeline(configs, pipeDir, automl=Settings.AUTOML_TPOT)
                    # Second part of config in not used
                    evalConf = [1, 0, 0, [[[], [], [clauseClass]], [[], []]], None, None, None, None]
                    # TODO try several vote
                    votes = [None]
                    for vote in votes:
                        f1, ev, pv = pt.evaluatePipeline(pipeDir, len(configs), evalConf, voteSent=None, voteClause=0, voteMode="not")

                        logging.info("Resulting f1: "+str(f1)+" with vote "+str(vote))

                        if f1 > f1ref:
                            f1ref = f1
                            selConfig = []
                            selvote = vote
                            selConfig.extend(configs)


        # Clear directory
        for f in os.listdir(pipeDir):
            os.remove(os.path.join(pipeDir, f))

        # Select best config
        logging.debug("\n"+clauseClass)
        logging.debug(f1ref)
        logging.debug(selConfig)
        pt.trainPipeline(selConfig, pipeDir, automl=Settings.AUTOML_TPOT)
        evalConf = [1, 0, 0, [ [ [], anoms, [clauseClass] ], [ [], [] ] ], None, None, None, None]
        res, ev, pv = pt.evaluatePipeline(pipeDir, len(selConfig), evalConf, voteSent=None, voteClause=selvote)
        logging.debug(res)
        logging.info(pt.generateClassifReport(ev, pv))

    # Default
    pipeDir = os.path.join(modelsPath, "default")
    # pipeDir = "Models\\pipeline\\phase3\\"+"default"
    if not os.path.exists(pipeDir):
        os.makedirs(pipeDir)
    transformers = ["count", "tfidf", "hash"]
    oversamplings = [False]
    allAnom = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15", "R16", "R17", "R18", "R19", "R20"]
    f1ref = 0
    selConfig = []

    for transformer in transformers:
        for oversampling in oversamplings:
            configs = []
            configs.append([1, 0, 0, [[[], allAnom, []], [[], []]], None, None, oversampling, transformer])
            # Sentence version TODO
            # configs.append([1, 1, 1, [ [ [], allAnom, [] ], [ [], allAnom ] ], None, None, oversampling, transformer])

            pt.trainPipeline(configs, pipeDir, automl=Settings.AUTOML_TPOT)
            # Second part of config in not used
            evalConf = [1, 0, 0, [[[], allAnom, []], [[], []]], None, None, None, None]
            # TODO try several vote
            f1, ev, pv = pt.evaluatePipeline(pipeDir, len(configs), evalConf, voteSent=1, voteClause=None)

            logging.info("Default: Resulting f1: "+str(f1)+" with vote "+str(vote))

            if f1 > f1ref:
                f1ref = f1
                selConfig = []
                selvote = vote
                selConfig.extend(configs)     

    # Clear directory
    for f in os.listdir(pipeDir):
        os.remove(os.path.join(pipeDir, f))

    # Select best config
    logging.debug("\n"+"Default")
    logging.debug(f1ref)
    logging.debug(selConfig)
    # TODO use previously used automl pipeline when using tpot
    pt.trainPipeline(selConfig, pipeDir, automl=Settings.AUTOML_TPOT)
    evalConf = [1, 0, 0, [[[], allAnom, []], [[], []]], None, None, None, None]
    res, ev, pv = pt.evaluatePipeline(pipeDir, len(selConfig), evalConf, voteSent=1, voteClause=None)
    logging.debug(res)
    logging.info(pt.generateClassifReport(ev, pv))

    
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)
    generatePhaseOnePipeline()
    generatePhaseTwoPipeline()
    generatePhaseThreePipeline()
