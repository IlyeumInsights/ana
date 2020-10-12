"""
Exploration of classical ML techniques.
Explore multiple feature extraction and ML classifier for clause classification.
It logs results in MLFlow and display visualization (facetgrid and heatmaps).
This file is relativaly volatile and is made to be changed a lot to experiment
according to ML techniques, feature extraction, data, etc...

:Authors: Nathan Ramoly
"""

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import SelectPercentile, f_classif, RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import logging

from time import gmtime, strftime

import mlflow
import mlflow.sklearn

from ana.preparation import DataPreparation
from ana.preparation import DataSelection
from ana.preparation import DataCleaning
from ana import Settings



def explorationProcess():

    # fes = ["bow", "tfidf", "hash", "doc2vec"]
    # mlc = ["lr", "ber", "lsvm", "svm", "rf", "knc", "ann"]
    fes = ["hash"]
    mlc = ["lsvm"]

    verbose = True

    currentTime = strftime("%Y-%m-%d_%H-%M-%S", gmtime())

    expeName = "beta_all_sent_"+currentTime

    mlflow.set_experiment(expeName)

    # For heatmaps
    df = pd.DataFrame(columns=['Classifier', 'Feature Extraction', 'A', 'P', 'R', 'F1'])
    # For facetgrid
    dfb = pd.DataFrame(columns=['Classifier', 'Feature Extraction', 'Metric', 'Value'])

    for fe in fes:

        x_train, _, y_train, _, _ = DataPreparation.prepareDataIly(
            0, 
            0,
            granularity=0,
            filter=[ [ [], [], [] ], [ [], [] ] ],
            labelBinary=True,
            oversample=False,
            transformer=fe,
            datasetFolder=Settings.TRAIN_DATASET_LOC,
            mappingExport=Settings.MODEL_MAP_DEFAULT)

        X, Y = DataSelection.loadData(
            [0,0], 
            filter=[ [ [], [], [] ], [ [], [] ] ], 
            granularity=0, 
            datasetFolder=Settings.TEST_DATASET_LOC)

        fX = []
        for id in X:
            fX.append(DataCleaning.clean(X[id]))
        X = np.array(fX)


        if fe != "doc2vec":
            vectorizer = pickle.load(open(Settings.MODEL_VECT_DEFAULT, "rb"))
            x_test = vectorizer.transform(X)
        else:
            # vectorizer = pickle.load(open( os.path.join(Settings.MODEL_DEST, "d2v.model"), "rb") )
            vectorizer = Doc2Vec.load(os.path.join(Settings.MODEL_DEST, "d2v.model"))
            x_test = []
            for text in X:
                x_test.append(vectorizer.infer_vector(text.split()))
            x_test = np.array(x_test)

        mapping = pickle.load(open(Settings.MODEL_MAP_DEFAULT, "rb"))
       
    
        expeVal = []
        for id in Y:
            if Y[id] in mapping:
                expeVal.append(mapping[Y[id]]) # mapping in range
            else:
                print("Unknown class"+str(Y[id]))
                expeVal.append(1)
        y_test = np.array(expeVal)
 
        # x_train, x_test, y_train, y_test, _ = DataPreparation.prepareDataIly(
        #     1, 
        #     0,
        #     granularity=0,
        #     filter=[ [ [], [], [] ], [ [], [] ] ],
        #     labelBinary=False,
        #     oversample=False,
        #     transformer=fe,
        #     datasetFolder=Settings.MAIN_DATASET_LOC)




        for classifier in mlc:

            with mlflow.start_run():

                mlflow.log_param("Feature Extraction", fe)
                mlflow.log_param("Classifier", classifier)

                if classifier == "lr":
                    mlmodel = LogisticRegression(C=5.0, dual=False, penalty='l2', verbose=verbose)
                elif classifier == "ber":
                    mlmodel = BernoulliNB(alpha=0.01, fit_prior=False)
                elif classifier == "lsvm":
                     mlmodel = make_pipeline(
                        RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=1.0, n_estimators=100), step=0.4),
                        LinearSVC(C=1.0, dual=False, loss="squared_hinge", penalty="l1", tol=0.01, verbose=verbose)
                    )
                elif classifier == "svm":
                    mlmodel =  SVC(gamma=2, C=1)
                elif classifier == "rf":
                    mlmodel = make_pipeline(
                        SelectPercentile(score_func=f_classif, percentile=66),
                        RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.2, min_samples_leaf=2, min_samples_split=2, n_estimators=100, verbose=verbose)
                    ) 
                elif classifier == "knc":
                    mlmodel = make_pipeline(
                        SelectPercentile(score_func=f_classif, percentile=45),
                        KNeighborsClassifier(n_neighbors=50, p=1, weights="distance")
                    )
                elif classifier == "ann":
                    if fe != "hash":
                        mlmodel = MLPClassifier(alpha=0.001, max_iter=100, verbose=verbose)
                    else: # Too long to compute, keeping only one case
                        break
                        # mlmodel = MLPClassifier(alpha=0.001, max_iter=1, verbose=verbose)
                else:
                    print("No model selected !!")
                    return 0

                print("Training with fe="+fe+" and ml="+classifier)

                # Training
                mlmodel.fit(x_train, y_train)

                # Prediction of the test set
                y_pred = mlmodel.predict(x_test)

                print("Predicted")
                print(y_pred)
                print("Expected:")
                print(y_test)

                print('Accuracy:%.2f%%' % (accuracy_score(y_test, y_pred)*100))
                print('Classification Report:')
                print(classification_report(y_test, y_pred))

                acc = accuracy_score(y_test, y_pred)
                pre = precision_score(y_test, y_pred, average="macro")
                rec = recall_score(y_test, y_pred, average="macro")
                f1 = f1_score(y_test, y_pred, average="macro")

                mlflow.log_metric("A", acc)
                mlflow.log_metric("P", pre)
                mlflow.log_metric("R", rec)
                mlflow.log_metric("F1", f1)

                row = [classifier, fe, acc, pre, rec, f1]
                df.loc[len(df)] = row

                row = [classifier, fe, "A", acc]
                dfb.loc[len(dfb)] = row
                row = [classifier, fe, "P", pre]
                dfb.loc[len(dfb)] = row
                row = [classifier, fe, "R", rec]
                dfb.loc[len(dfb)] = row
                row = [classifier, fe, "F1", f1]
                dfb.loc[len(dfb)] = row

                mlflow.end_run()
            # end mflow.run
        # end for class
    # end for fe

    # Saving
    csvFacet = f"mlruns\\data\\"+expeName+"_dff.csv"
    csvHeatM = f"mlruns\\data\\"+expeName+"_dfh.csv"
    dfb.to_csv(csvFacet)
    df.to_csv(csvHeatM)

    # g = sns.catplot(x='Feature Extraction', y='Value', hue='Metric', row='Classifier', data=dfb, legend=True, kind="bar")
    g = sns.FacetGrid(dfb, row="Classifier", col="Feature Extraction", margin_titles=True )
    g.set(ylim=(0.0, 1.0))
    g = (g.map(sns.barplot, "Metric", "Value", palette='Set1')).add_legend()
    # for ax in g.axes.flatten():
    #     ax.set_ylabel('')
    #     ax.set_xlabel('')

    plt.show()
    pngFacet = f"mlruns\\data\\"+expeName+"_dff.png"
    g.savefig(pngFacet)

    df2 = df.pivot("Feature Extraction", "Classifier", "F1")
    ax = sns.heatmap(df2, annot=True, vmin=0.0, vmax=1)

    plt.show()
    pngHeatM = f"mlruns\\data\\"+expeName+"_dfh.png"
    # ax.savefig(pngHeatM)

    # mlflow.log_artifact(csvFacet)
    # mlflow.log_artifact(pngFacet)
    # mlflow.log_artifact(csvHeatM)
    # mlflow.log_artifact(pngHeatM)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)
    explorationProcess()