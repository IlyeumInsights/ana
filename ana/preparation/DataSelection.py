"""Module DataSelection
*****************************

Module in charge of selecting data according to the different level of
annotation (labels) based on a filter based query.
It allows to select data of a particular type or anomalies.
It also aggregates lower granules (sentences) to bigger one (clause).

Functions
    loadData
        Load the text data from a dataset location.


:Authors: Nathan Ramoly
"""

import csv
import os
import logging

from ana import Settings

def loadData(colY, filter=[[[], [], []], [[], []]], granularity=0,
             datasetFolder=Settings.MAIN_DATASET_LOC):
    """loadData

    Read and load a dataset folder that contains annotation files (csv) and
    a subdirectories with the raw texts.

    :param colY: The column that should be used for classification (the labels
        or Y) as: [granularity, column] both as int (pos)
    :type colY: list((int, int))
    :param filter: The list, for each granularity level and each column, of
        accepted classes (if empty, all class are accepted) granularity is the
        granulirity that is to be considered (0 highest, ex: clause), if the colY
        is from a higher granularity, annotation are propagated to children.
    :type filter: list(list(list(str)))
    :param granularity: Granulirty level used for loading (sentence, clause...),
        default: 0. Is the higher level, other values are deeper/smaller text
        decomposition.
    :type granularity: int
    :param datasetFolder: Location of the dataset directory.
    :type datasetFolder: str (dirPath)

    :return: The set of values/texts (X).
    :rtype: dict(str, str)
    :return: The set of associated label (Y).
    :rtype: dict(str, str)
    """
    labels = loadLabels(datasetFolder)

    flabels = filterLabels(labels, filter)
    # logging.debug(flabels)

    data = loadText(datasetFolder, flabels)
    # logging.debug()
    # logging.debug(data)

    X, Y = selectLearningClass(colY, granularity, data, flabels)

    # logging.debug("---------------------------------------------------")
    # logging.debug(Y)
    # logging.debug(X)

    return X, Y


def loadLabels(datasetFolder):
    """loadLabels

    Loard the labels with the following format:
    Format [dictA, dictB, dictC, ...]
    Granule level: ...C in B in A
    Ids: idA, idA_idB, idA_idB_idC
    dict: {id, [col]}

    For now, it uses default annotatrion csv files.

    :param datasetFolder: Location of the dataset directory.
    :type datasetFolder: str (dirPath)

    :return: List of labels decomposed by granularity and dimensions (class
        domain).
    :rtype: list(dict(str, list(str)))
    """
    # Files by granularity order
    fileNames = Settings.DATASET_LABEL_FILES

    labels = []

    delim = ","
    for fileName in fileNames:
        newDict = {}
        with open(os.path.join(datasetFolder, fileName), 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=delim)
            for row in csv_reader:
                if len(row) < 2:
                    logging.error("Misformated row in classes.csv !")
                else:
                    newDict[row[0]] = []
                    for i in range(1, len(row)):
                        newDict[row[0]].append(row[i].strip())
        labels.append(newDict)
    return labels

def filterLabels(labels, selections):
    """filterLabels

    Selects labels according to this template:
    [ [ [classesA_co1], [classesA_co2] ], [ [classesB_co1], [classesB_co2] ] ]
    (See loadData explication for filter)

    :param labels: List of labels decomposed by granularity and dimensions.
    :type labels: list(dict(str, list(str)))

    :param selections: Filter query.
    :type selections: list(list(list(str)))

    :return: The set of laabels matching the filter.
    :rtype: list(dict(str, list(str))) (to check)
    """
    if len(labels) != len(selections):
        logging.error("Selection data do not match labels list !")

    flabels = []

    for i in range(0, len(labels)):
        label = labels[i]
        selec = selections[i]

        flabel = selectLabel(label, selec)

        flabels.append(flabel)

    flabels = recursiveSelect(flabels)

    return flabels


def selectLabel(label, selection):
    """selectLabel

    Apply selection on a single label list, that is the labels for one entry
    for each dimension/domain and granularity

    :param label: Labels for one entry/segment.
    :type label: dict(str, list(str))

    :param selections: Filter query.
    :type selections: list(list(list(str)))

    :return: Label that fits the filter query
    :rtype: dict(str, list(str)) (to check)
    """
    # Filtered label
    flabel = {}
    for id in label:
        filtered = False
        # Go through labels associated to a granule
        for i in range(0, len(label[id])):
            if not (label[id][i] in selection[i] or len(selection[i]) <= 0):
                filtered = True
        if not filtered:
            flabel[id] = label[id]
    return flabel


def recursiveSelect(labels):
    """recursiveSelect

    Filter labels of child granules recurisvely

    :param labels: List of labels decomposed by granularity and dimensions.
    :type labels: list(dict(str, list(str)))

    :return: filtered label
    """
    # Go by pair
    for i in range(0, len(labels)-1):
        # Get ids of parent
        idsParent = []
        # Transform keys
        for idkey in labels[i].keys():
            idsParent.append(int(idkey))

        # Going through keys of child
        newLabel = {}
        for keychild in labels[i+1].keys():
            # Get parent part from key (format idA_idB_idC)
            # thus i matches A,B,C...
            idparInKeychild = int(keychild.split("_")[i])
            if idparInKeychild in idsParent:
                newLabel[keychild] = labels[i+1][keychild]
                # del labels[i+1][keychild]
        labels[i+1] = newLabel
    return labels


def loadText(datasetFolder, labels):
    """loadText

    Load the raw text according to the selected labels through their ids.

    :param datasetFolder: Location of the dataset directory.
    :type datasetFolder: str (dirPath)

    :param labels: List of labels per id.
    :type labels: list(dict(str, list(str)))

    :return: Selected raw text
    :rtype: dict(str, str)
    """
    # Get selected id for lower level labels
    ids = labels[-1].keys()

    textFolder = os.path.join(datasetFolder, "texts")

    texts = {}

    for idfile in ids:
        filename = idfile+".txt"
        with open(os.path.join(textFolder, filename), encoding='utf8') as textFile:
            texts[idfile] = textFile.read()

    return texts

def mergeGranule(data):
    """mergeGranule

    Merges granules to the upper level.
    For now, only for higher level (clause).
    Example: sentences to clause

    :param data: Textual subgranules.
    :type data: dict(str, str)

    :return: Text for upper level granule.
    :rtype: dict(str, str)

    """
    # intermediate List of subgranule per granule
    intList = {}
    # FInal text
    granuleTexts = {}

    # list all subgranule, then order them before generation of text
    for subgranule in data:
        granuleid = subgranule.split("_")[0] # Only first/higher for now
        if granuleid not in intList:
            intList[granuleid] = []
        intList[granuleid].append(subgranule)

    for granId in intList:
        intList[granId].sort()
        # Generate
        granText = ""
        for subtextId in intList[granId]:
            granText += data[subtextId]+" "
        granuleTexts[granId] = granText

    return granuleTexts



def selectLearningClass(colY, granularity, data, labels):
    """selectLearningClass

    Determine X and Y
    .. todo:: diff granularity level

    :param colY: The column that should be used for classification (the labels
        or Y) as: [granularity, column] both as int (pos)
    :type colY: list((int, int))
    :param granularity: Granulirty level used for loading (sentence, clause...).
    :type granularity: int

    :param data: Textual granules.
    :type data: dict(str, str)

    :param labels: List of labels per id.
    :type labels: list(dict(str, list(str)))

    :return: The set of values/texts (X).
    :rtype: dict(str, str)
    :return: The set of associated label (Y).
    :rtype: dict(str, str)
    """
    X = {}
    Y = {}

    if granularity < colY[0]: # Large granule but precise annotation
        logging.error("Trying to annotate a high level granule with lover level annotation !")
        return None, None
    elif granularity > colY[0]: # Precise granule but broad annotation
        # propagate
        propagate = True
    else:
        propagate = False

    if granularity < 1:
        data = mergeGranule(data)

    dictLabel = labels[colY[0]] # colY[0] is the gran level

    for idgran in data:
        if propagate:
            # Get id of higer level granule
            idLabel = idgran.split("_")[0]
        else:
            idLabel = idgran

        if idLabel in dictLabel:
            X[idgran] = (data[idgran])
            Y[idgran] = (dictLabel[idLabel][colY[1]])
        else:
            logging.error("Unknown key when generating X and Y: "+idgran+" "+idLabel)

    return X, Y
