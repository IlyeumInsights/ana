"""Module CompositeClassifier
*****************************

Module created in an effort to simplify the structure of the code.
Work in progress. (Currently not used)

Class:
    CompositeClassifier: Classifier composed of several complementary
    subclassifiers.

.. todo:: use composite classifier for all case (would match ensemble classifier)
.. todo:: Move in a suitable package

:Authors: Nathan Ramoly
"""
import logging

from sklearn.base import BaseEstimator, ClassifierMixin


class CompositeClassifier(BaseEstimator, ClassifierMixin):
    """ class CompositeClassifier
    aka Reduced Ensemble Classifier

    Classifier composed of several complementary subclassifiers.
    This classifier is typically a multiclass classifier decomposed in binary
    subclassifiers.
    Classification is then agregated with all detected class.
    For now, agregation is only working for binary subclassifier.
    """

    def __init__(self, models=[]):
        self.models = models

    def fit(self, X, y=None):
        """fit method
        Train all subclassifiers.
        """
        for model in self.models:
            model.fit(X, y)

    def predict(self, X, y=None):
        """predict method
        Evaluate X from all classifier.

        :return: Classification for each value in X.
        """
        if len(self.models) <= 0:
            logging.warning("No subclassifier in Composite Classifier object.")
        ys = []
        for model in self.models:
            ys.append(model.predict(X))
        cy = self._aggregateY(ys, X)
        return cy        

    def _aggregateY(self, ys, X):
        """
        Return result per X (on array per value in X) from arrays per 
        subclassifier.
        """
        cy=[] # composite y
        for i in range(0,len(X)):
            cy[i] = []
            for y in ys:
                cy[i].append(y[i])
        return cy

    def score(self, X, y=None):
        """ score method
        ..todo:
        """
        return
