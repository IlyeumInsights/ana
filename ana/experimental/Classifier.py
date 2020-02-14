"""
Class that load trained model and use them to perform classification
Used for debug/testing only !
"""
import pickle
from joblib import load
import numpy as np

from ana.preparation import DataCleaning, DataPreparation

class Classifier:

    def __init__(self):
        self.vectorizer = pickle.load( open("models\\vectorizer.pickle","rb") )
        self.nmodel = load('models\\model.joblib')

    def classify(self, text):
        text = DataCleaning.clean(text)
        textnp = np.array([text])
        vectext = self.vectorizer.transform(textnp)
        y = self.nmodel.predict(vectext)

        return y

if __name__ == '__main__':  
    clf = Classifier()

    text = ""
    with open("some_path_to_text", encoding="utf8") as toRead:
        text = toRead.read()
    print(text)

    print( clf.classify(text) )

    # Full directory classification
    segments, labels = DataPreparation.loadDataset("some_path_to_a_dir")

    for i in segments:
        print("--------")
        print(i)
        print(segments[i])
        print(labels[i])
        print( clf.classify(segments[i]))
