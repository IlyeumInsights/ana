"""Module DataPreparation
*****************************

Provides function to Clean, Transform and Format data in preparation for the ML.

Functions:
    loadDataset
        Loads the raw data from a directory.
    transform
        Transforms texts into features thanks to textual feature extraction
        technique including bow (tf-idf, count, hashing) and Doc2Dev.
    prepareData
        Load dataset and extract features to prepare data for ML.
    prepareDataIly
        Same as prepareData, but specifically on Ilyeum contract dataset.
    preparationReport
        Provides a textual report of the complete preparation process.

:Authors: Nathan Ramoly
"""

import csv
import os
import logging
import numpy as np
import pickle
from random import randrange

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.manifold import TSNE
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

from ana.preparation import DataCleaning, DataSelection
from ana.preparation.TextGeneticOversampler import TextGeneticOversampler
from ana import Settings

def loadDataset(datadir=Settings.MAIN_DATASET_LOC):
    """loadDataset

    Generates two dictionnaries describing segments from a directory of a
    dataset containing raw data and annotation.
    .. todo:: Handle default directory pass.

    :param datadir: directory of the dataset to load.
    :type text: str (filepath)

    :return: 
        - Dictionnary associating segment to text (segId, text)
        - Dictionnary associating segment to label/annotation (segId, label)
    :rtype: dict(int, str), dict(int, str)

    """

    segments = {}
    labels = {}

    # Reading the classes CSV
    delim = ","
    with open(os.path.join(datadir, "classes.csv"), 'r') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=delim)
        for row in csvReader:
            if len(row) < 2:
                logging.error("Misformated row in classes.csv !")
            else:
                labels[row[0]] = row[1].strip()

    # Reading texts
    textDir = os.path.join(datadir, "texts")
    for textFileName in os.listdir(textDir):
        # logging.debug(textFileName)
        with open(os.path.join(textDir, textFileName), encoding='utf8') as textFile:
            segments[os.path.splitext(textFileName)[0]] = textFile.read()

    if len(segments) != len(labels):
        logging.error("Mismatching number of segments and labels: "+str(len(segments))+" <> "+str(len(labels)))

    return segments, labels

def cleanSegments(segments):
    """cleanSegments

    Clean all segments using the DataCleaning module.

    :param segments: Raw segments of text.
    :type segments: dict(int, str)

    :return: Clean segments.
    :rtype: dict(int, str)
    """
    newSegList = {}
    for id in segments:
        cleanSeg = DataCleaning.clean(segments[id])
        newSegList[id] = cleanSeg
    return newSegList

def convertLabels(labels, classes, binary=False, refClass=None):
    """convertLabels

    Convert the string label to numerical one.
    Binary = if different than one class
    refClass = list of reference classes

    :param labels: Dictionnary of raw annotation to convert per segment id.
    :type labels: dict(str, str)

    :param classes: List of all known classes in the domain.
    :type classes: list(str)

    :param Binary: Convert label into 1 and 0 values, 1 meaning the label
        matches the reference class, 0 otherwise. The reference classes are provided
        or, by default, is the first class in the 'classes' array.
    :type Binary: bool

    :param refClass: Reference classes that are to be converted, while the
        other ones are ignored.
    :type refClass: list(str)

    :return: Set of label/class as numerical values.
    :rtype: dict(str, int)
    :return: Mapping between textual and numerical labels.
    :rtype: dict(str, int)
    """
    numLab = {}
    mapping = {}
    for id in labels:
        if binary:
            if labels[id] != classes[0]:
                numLab[id] = 1
            else:
                numLab[id] = -1
        elif refClass is not None:
            if labels[id] in refClass:
                numLab[id] = refClass.index(labels[id])+1 # Start from 1
            else:
                numLab[id] = -1
        else:
            numLab[id] = classes.index(labels[id])
        mapping[labels[id]] = numLab[id]
    return numLab, mapping

def transform(segments, labels, transformer="count", oversample=False,
              trasnformSerial=Settings.MODEL_VECT_DEFAULT):
    """transform

    Transform a set of textual segments into features.
    It offer multiple feature extraction techniques and can oversample the
    data set classicaly or geneticaly (see TextGeneticOversampler module).

    :param segments: Raw segments of text.
    :type segments: list(str)

    :param labels: List of raw annotation associated to each segment.
    :type oversample: list(str)

    :param transformer:
        - 'count' or 'bow' (default): standard BOW using sklearn.CountVectorizer.
        - 'tfidf' or 'tf-idf': TF-IDF feature extraction through sklearn.TfidfVectorizer.
        - 'hash':  Uses sklearn.HashingVectorizer
        - 'doc2vec': DL doc2vec approach provided by Gensim.
    :type transformer: str

    :param oversample: Assert if oversample should be performed or not.
    :type oversample: bool

    :param trasnformSerial: Transform process export path.
    :type trasnformSerial: str (filepath)

    :return:
        - Set of data of training (x_train)
        - Set of label of training (y_train)
        - Set of data of testing (x_test)
        - Set of label of testing (y_test)
    :rtype: np.array
    """
    npseglist = []
    nplablist = []

    for id in segments:
        npseglist.append(segments[id])
        nplablist.append(labels[id])

    npseg = np.array(npseglist)
    nglab = np.array(nplablist)
    
    # Separate test and train (work on numpy, can only nbe performed in tansform)
    npseg_train, npseg_test, nglab_train, nglab_test = train_test_split(npseg, nglab, test_size = 0.25, random_state = 1)

    # Debug when split is perfomred manually beforehand
    npseg_train = npseg
    nglab_train = nglab

    # logging.debug(len(npseg_train))
    if oversample:
        tgo = TextGeneticOversampler(generation=3)
        npseg_train, nglab_train = tgo.fit_resample(npseg_train, nglab_train)
    # logging.debug(len(npseg_train))
    # logging.debug(np.unique(nglab_train, return_counts=True)[0])
    # logging.debug(np.unique(nglab_train, return_counts=True)[1])

    if transformer == "tfidf" or transformer == "tf-idf":
        x_train, y_train, x_test, y_test = transformTFIDF(npseg_train, nglab_train, npseg_test, nglab_test, trasnformSerial)
    elif transformer == "hash":
        x_train, y_train, x_test, y_test = transformHashing(npseg_train, nglab_train, npseg_test, nglab_test, trasnformSerial)
    elif transformer == "count" or transformer == "bow":
        x_train, y_train, x_test, y_test = transformBOW(npseg_train, nglab_train, npseg_test, nglab_test, trasnformSerial)
    elif transformer == "doc2vec":
        x_train, y_train, x_test, y_test = transformDoc2Vec(npseg_train, nglab_train, npseg_test, nglab_test)
    else:
        x_train, y_train, x_test, y_test = transformBOW(npseg_train, nglab_train, npseg_test, nglab_test, trasnformSerial)
    

    x_train, y_train = oversampling(x_train, y_train)

    return x_train, x_test, y_train, y_test


def transformBOW(segments, labels, segments_test=None, labels_test=None,
                 serialPath=Settings.MODEL_VECT_DEFAULT):
    """transformBOW

    Apply feature extraction with sklearn.CountVectorizer.
    It relies on char_wb and an ngram range of [4,5].
    .. todo:: Adapt parameter in the AutoML process

    :param segments: Raw segments of text.
    :type segments: np.array

    :param labels: List of raw annotation associated to each segment.
    :type labels: np.array

    :param serialPath: Transform process export path.
    :type serialPath: str (filepath)

    :param segments_test: Test segments to transform.
    :type segments_test: np.array

    :param labels_test: List of labels on the test set.
    :type labels_test: np.array


    :return:
        - Set of data of training (x_train)
        - Set of label of training (y_train)
        - Set of data of testing (x_test)
        - Set of label of testing (y_test)
    :rtype: np.array
    """
    # textVocabPay = ["chèque", "cheque", "virement", "15", "30", "45", "60", 
    #                 "quinze", "trente", "quarante cinq", "soixante"]
    
    vectorizer = CountVectorizer(min_df=1, analyzer='char_wb', ngram_range=(4, 5))
    # vectorizer = CountVectorizer(min_df=1, vocabulary=textVocabPay)
    vectors = []
    vectors = vectorizer.fit_transform(segments)
    pickle.dump(vectorizer, open(serialPath, "wb"))

    vectors_test = None
    if segments_test is not None and labels_test is not None:
        vectors_test = vectorizer.transform(segments_test)

    # logging.debug(vectorizer.get_feature_names())

    return vectors, labels, vectors_test, labels_test


def transformHashing(segments, labels, segments_test=None, labels_test=None, 
                     serialPath=Settings.MODEL_VECT_DEFAULT):
    """transformHashing

    Apply feature extraction with sklearn.HashingVectorizer.

    :param segments: Raw segments of text.
    :type segments: np.array

    :param labels: List of raw annotation associated to each segment.
    :type labels: np.array

    :param serialPath: Transform process export path.
    :type serialPath: str (filepath)

    :param segments_test: Test segments to transform.
    :type segments_test: np.array

    :param labels_test: List of labels on the test set.
    :type labels_test: np.array

    :return:
        - Set of data of training (x_train)
        - Set of label of training (y_train)
        - Set of data of testing (x_test)
        - Set of label of testing (y_test)
    :rtype: np.array
    """
    vectorizer = HashingVectorizer()
    vectors = []
    vectors = vectorizer.fit_transform(segments)
    pickle.dump(vectorizer, open(serialPath, "wb"))

    vectors_test = None
    if segments_test is not None and labels_test is not None:
        vectors_test = vectorizer.transform(segments_test)

    return vectors, labels, vectors_test, labels_test


def transformTFIDF(segments, labels, segments_test=None, labels_test=None,
                   serialPath=Settings.MODEL_VECT_DEFAULT):
    """transformTFIDF

    Apply feature extraction with sklearn.TfidfVectorizer.

    :param segments: Raw segments of text.
    :type segments: np.array

    :param labels: List of raw annotation associated to each segment.
    :type labels: np.array

    :param serialPath: Transform process export path.
    :type serialPath: str (filepath)

    :param segments_test: Test segments to transform.
    :type segments_test: np.array

    :param labels_test: List of labels on the test set.
    :type labels_test: np.array

    :return:
        - Set of data of training (x_train)
        - Set of label of training (y_train)
        - Set of data of testing (x_test)
        - Set of label of testing (y_test)
    :rtype: np.array
    """
    vectorizer = TfidfVectorizer()
    vectors = []
    vectors = vectorizer.fit_transform(segments)
    pickle.dump(vectorizer, open(serialPath, "wb"))

    # logging.debug(vectorizer.get_feature_names())
    vectors_test = None
    if segments_test is not None and labels_test is not None:
        vectors_test = vectorizer.transform(segments_test)

    # display_scores(vectorizer, vectors)

    return vectors, labels, vectors_test, labels_test


def transformDoc2Vec(segments, labels, segmentsTest=None, labelsTest=None):
    """transformDoc2Vec

    Apply feature extraction with gensim.Doc2Vec.
    Be aware that this process is heavy.

    :param segments: Raw segments of text.
    :type segments: np.array

    :param labels: List of raw annotation associated to each segment.
    :type labels: np.array

    :param segmentsTest: Test segments to transform.
    :type segmentsTest: np.array

    :param labelsTest: List of labels on the test set.
    :type labelsTest: np.array

    :return:
        - Set of data of training (x_train)
        - Set of label of training (y_train)
        - Set of data of testing (x_test)
        - Set of label of testing (y_test)
    :rtype: np.array
    """
    labsegments = [TaggedDocument(doc, [i]) for i, doc in enumerate(segments)]

    model = Doc2Vec(vector_size=30, window=2, min_count=2, workers=6, dm =1, epochs=10)
    model.build_vocab(labsegments)
    model.train(labsegments, total_examples=model.corpus_count, epochs=model.iter)

    path = os.path.join(Settings.MODEL_DEST, "d2v.model")
    model.save(path)

    vectors = []
    for segment in segments:
        vectors.append(model.infer_vector(segment.split()))

    vectorsTest = None
    if segmentsTest is not None and labelsTest is not None:
        vectorsTest = []
        for segment in segmentsTest:
            vectorsTest.append(model.infer_vector(segment.split()))

    vectors = np.array(vectors)
    vectorsTest = np.array(vectorsTest)

    debug_tsne_plot(model)

    return vectors, labels, vectorsTest, labelsTest


def oversampling(X, Y, random=0):
    """oversampling

    Oversample datasets (already transformed)
    The resulting size of each class set is almost equal.

    :param X: Data to oversample.
    :type X: np.array

    :param Y: Labels associated to the data.
    :type Y: np.array
    """
    ros = RandomOverSampler(random_state=random)
    X, Y = ros.fit_resample(X, Y)

    return X, Y

def classFilter(segments, labels, classes):
    """ classFilter

    Filter the dataset with only selected classes.

    :param segments: Segments of text to filter.
    :type segments: dict(int, str)

    :param labels: Dictionnary of raw annotation to convert per segment id.
    :type labels: dict(str, str)

    :param classes: List of all accepted class.
    :type classes: list(str)


    :return: Filtered set of segments of text.
    :rtype: dict(int, str)
    :return: Filtered labels.
    :rtype: dict(str, str)
    """
    filteredSegments  = {}
    filteredLabels = {}

    for id in segments:
        if labels[id] in classes:
            filteredSegments[id] = segments[id]
            filteredLabels[id] = labels[id]

    return filteredSegments, filteredLabels


def debug_tsne_plot(model):
    """ debug_tsne_plot

    Creates and TSNE model and plots it.
    Used for deub/experiments.
    From https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne
    """
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        # plt.annotate(labels[i],
        #              xy=(x[i], y[i]),
        #              xytext=(5, 2),
        #              textcoords='offset points',
        #              ha='right',
        #              va='bottom')
    plt.show()

def display_scores(vectorizer, tfidf_result):
    """display_scores

    Debug function.
    http://stackoverflow.com/questions/16078015/
    """
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    for item in sorted_scores:
        logging.info("{0:50} Score: {1}".format(item[0], item[1]))

def prepareData():
    """prepareData

    Load, clean and transform a dataset from a fixed location
    It povides the data and annotation ready to be processed my ML tools.
    Not used anaymore.
    .. todo:: Perform data load from a database and a selected domain.

    :return: Set of data of training (x_train)
    :rtype: np.array
    :return: Set of label of training (y_train)
    :rtype: np.array
    :return: Set of data of testing (x_test)
    :rtype: np.array
    :return: Set of label of testing (y_test)
    :rtype: np.array
    """
    classes = ["None", "Non_sollicitation", "Resiliation", "Indemnisation", "Paiement", "Autres"]
    # classes = ["None","False","Missing","Imprecise"]
    # classes = ["none","delai","non_reciprocite","montant","indemnisation","contact","autres","type_paiement","condition_paiement"]

    segments, labels = loadDataset(Settings.MAIN_DATASET_LOC)

    # segments, labels = classFilter(segments, labels, [classes[0], classes[2]])

    segments = cleanSegments(segments)
    labels = convertLabels(labels, classes, True, None)

    x_train, x_test, y_train, y_test = transform(segments, labels)

    # X, Y = oversample(X, Y, 1)

    return x_train, x_test, y_train, y_test



def prepareDataIly(labelType, labelLevel, granularity=0, 
                   filter=[[[], [], []], [[], []]],
                   labelBinary=False, labelClasses=None, oversample=False,
                   transformer="count",
                   datasetFolder=Settings.MAIN_DATASET_LOC,
                   trasnformSerial=Settings.MODEL_VECT_DEFAULT,
                   mappingExport=None):
    """prepareDataIly

    Load, clean and transform Ilyeum contract dataset?
    It povides the data and annotation ready to be processed my ML tools.
    .. todo:: Perform data load from a database and a selected domain.

    :return: Set of data of training (x_train)
    :rtype: np.array
    :return: Set of label of training (y_train)
    :rtype: np.array
    :return: Set of data of testing (x_test)
    :rtype: np.array
    :return: Set of label of testing (y_test)
    :rtype: np.array
    """
    classes1 = ["None", "False", "Missing", "Imprecise"]
    # classes2 = ["none","delai","non_reciprocite","montant","indemnisation","contact","autres","type_paiement","condition_paiement"]
    classes2 = ["None", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15", "R16", "R17", "R18", "R19", "R20"]
    classes3 = ["objet", "general", "duree", "lieu", "obligations", "paiement", "resiliation", "non_sollicitation", "non_concurrence", "confidentialite", "responsabilites", "indemnisation", "prop_intel", "legal", "autres"]

    classes = [classes1, classes2, classes3]

    segments, labels = DataSelection.loadData([labelLevel, labelType], filter, granularity, datasetFolder)

    segments = cleanSegments(segments)
    labels, mapping = convertLabels(labels, classes[labelType], labelBinary, labelClasses)

    #Save mapping
    if mappingExport is not None:
        pickle.dump(mapping, open(mappingExport, "wb"))

    x_train, x_test, y_train, y_test = transform(segments, labels, transformer=transformer, oversample=oversample, trasnformSerial=trasnformSerial)

    # x_train, y_train = oversampling(x_train, y_train)

    report = preparationReport(labelType, labelLevel, granularity, filter,
                               labelBinary, labelClasses, oversample,
                               datasetFolder, classes[labelType],
                               labels, x_train, x_test, y_train, y_test,
                               mapping, transformer)
    # logging.debug(report)

    return x_train, x_test, y_train, y_test, report



def preparationReport(labelType, labelLevel, granularity, filter, labelBinary, labelClasses, oversample, datasetFolder, classes, labels, x_train, x_test, y_train, y_test, mapping, transformer):
    """preparationReport

    Generate a string report of the data preparation.

    :return: Report of preparation of data.
    :rtype: str
    """
    report = "\nDATA PREPARATION REPORT: \n"
    report += "-----\n"

    report += "Data source:        "+datasetFolder+"\n"
    report += "Data Granularity:   "+str(granularity)+"\n"
    report += "Label Granularity:  "+str(labelLevel)+"\n"
    report += "Label classif col:  "+str(labelType)+"\n"
    report += "Label classif cla:  "+str(classes)+"\n"
    report += "Data selection:     "+str(filter)+"\n"
    report += "Selected Transform: "+transformer+"\n"
    report += "Binary classif:     "+str(labelBinary)+"\n"
    report += "Restricted classif: "+str(labelClasses)+"\n"
    report += "Genetic oversample: "+str(oversample)+"\n"

    report += "-----\n"

    report += "Labels Mapping: "+"\n"
    for lab in mapping:
        report += str(lab)+": "+str(mapping[lab])+"\n"

    report += "-----\n"

    report += "Initial Y size: "+str(len(labels))+"\n"
    report += "post-transform X train size: "+str((x_train.shape[0]))+"\n"
    report += "post-transform Y train size: "+str((y_train.shape[0]))+"\n"
    report += "post-transform X test size:  "+str((x_test.shape[0]))+"\n"
    report += "post-transform Y test size:  "+str((y_test.shape[0]))+"\n"

    report += "-----\n"
    report += "END DATA PREPARATION REPORT. \n"

    return report
