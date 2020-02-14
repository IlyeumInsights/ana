"""
Testing ML approacher
"""

from sklearn.svm import LinearSVC, OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_selection import SelectPercentile, f_classif, RFE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import classification_report

import mlflow
import mlflow.sklearn

# import sys
# sys.path.insert(1, 'C:\\Users\\Nara\\Workspace\\python\\contract-analysis\\')
from ana.preparation import DataPreparation

# x_train, x_test, y_train, y_test = DataPreparation.prepareData()

# Granule/labelevel 0 = clause (higher) 1 sentence
x_train, x_test, y_train, y_test, dataprepReport = DataPreparation.prepareDataIly(1, 0, granularity=1, filter=[ [ [], [], ["resiliation"] ], [ [], [] ] ], labelBinary=True, oversample=False, transformer="tfidf")

# print(X.toarray())
# print(len(y_train))
# print(Y)

print("Training set size: ")
print(y_train.shape[0])

# x_train, y_train = DataPreparation.oversample(x_train, y_train)

# print("Training set size after oversample: ")
# print(len(y_train))
with mlflow.start_run():
    # mlmodel = LinearSVC(verbose=0)
    # mlmodel =  BernoulliNB(alpha=0.01, fit_prior=False)()
    # mlmodel = LogisticRegression(C=5.0, dual=True, penalty='l2')
    # mlmodel = exported_pipeline = make_pipeline(
    #     SelectPercentile(score_func=f_classif, percentile=45),
    #     KNeighborsClassifier(n_neighbors=91, p=1, weights="distance")
    # )
    # mlmodel = make_pipeline(
    #     SelectPercentile(score_func=f_classif, percentile=66),
    #     RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.2, min_samples_leaf=2, min_samples_split=2, n_estimators=100)
    # )
    # mlmodel = make_pipeline(
    #     RFE(estimator=ExtraTreesClassifier(criterion="entropy", max_features=0.4, n_estimators=100), step=0.3),
    #     KNeighborsClassifier(n_neighbors=58, p=2, weights="distance")
    # )
    # mlmodel = make_pipeline(
    #     RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=1.0, n_estimators=100), step=0.4),
    #     LinearSVC(C=1.0, dual=False, loss="squared_hinge", penalty="l1", tol=0.01)
    # )
    C = 25.0
    tol = 0.001
    mlmodel = LinearSVC(C=C, dual=False, loss="squared_hinge", penalty="l2", tol=tol)
    mlmodel.fit(x_train, y_train) # y not used in oneclassSVM 

    # mlmodel = OneClassSVM(kernel="rbf")
    # mlmodel.fit(x_train)

    # Export

    from joblib import dump
    dump(mlmodel, 'Models\\model.joblib')

    y_pred = mlmodel.predict(x_test)


    # Validation
    # from sklearn.model_selection import cross_val_score

    # print( cross_val_score(mlmodel, X, Y, cv=50, scoring='f1_macro') )

    print("Predicted")
    print(y_pred)
    print("Expected:")
    print(y_test)

    print('Accuracy:%.2f%%' % (accuracy_score(y_test, y_pred)*100))
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    
    mlflow.log_param("C", C)
    mlflow.log_param("tol", tol)
    mlflow.log_metric("acc", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1", f1_score(y_test, y_pred))
    mlflow.log_metric("rec", recall_score(y_test, y_pred))
    mlflow.log_metric("prec", precision_score(y_test, y_pred))
    mlflow.sklearn.log_model(mlmodel, "models_anom_resiliation")


# TPOT

# from tpot import TPOTClassifier

# tpot = TPOTClassifier(generations=10, population_size=100, cv=5, n_jobs=6, verbosity=2, scoring='f1_macro', config_dict='TPOT sparse')
# tpot.fit(x_train, y_train)

# print(tpot.score(x_test, y_test))

# y_pred = tpot.predict(x_test)
# print(classification_report(y_test, y_pred))

# tpot.export('tpot_export.py')
