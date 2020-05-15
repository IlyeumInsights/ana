"""
Restful Flask-based server providing API to call ANA for classification.
This API allows for now to send a text (with title and domain)
and provides back the classification for the each phase alpha, beta and gamma.
More task, including getting the available domains or previous classification
will be added later.

:Authors: Nathan Ramoly
"""
import logging
import os
import sys

from flask import Flask, request, jsonify
from flask_restful import abort

# Launching the Flask APP
# NOTE: this should be perfom before immporting project modules
PROJECT_HOME = os.getcwd()
if PROJECT_HOME not in sys.path:
    sys.path = [PROJECT_HOME] + sys.path

APP = Flask(__name__)

from ana.evaluation import ClauseAnalyzer


@APP.route('/api/classsify', methods=['POST'])
def classifyText():
    """classifyText

    Classifies a text (supposedly a clause) provided by HTTP post in JSON.
    It performs ANA classification to infer the type, if the clause
    is anomalous and the type of the anomaly.

    The expected content of the input json are:
    * title: the title of the clause.
    * text: the textual content of the clause.
    * domain: the domaine the clause belong to (ex: ToS or service contracts).

    :return: Answer json including: title, type, anomalous and anomaly.
    :rtype: json
    """

    if not request.json or \
        not 'text' in request.json or \
        not 'title' in request.json or \
        not 'domain' in request.json:
        logging.error("Post message wrongly formatted !")
        abort(400)
    else:
        logging.info("Classifiying and generating answer...")

        message = request.json

        cType, isAnom, aType = ClauseAnalyzer.analyzeClause(message['text'], message['title'])

        answer = {
            'title': message['title'],
            'type': str(cType),
            'anomalous': str(isAnom),
            'anomaly': aType
        }
        return jsonify(answer), 201


@APP.route('/api/classsify_dynamic', methods=['POST'])
def dynamicClassifyText():
    """ dynamicClassifyText

    Perform clause analysis according to the provided information.
    It uses available data to shortcut the analysis if possible.
    Typically, if no data is given, all phases are performed.
    On the other hand, if the type and abnormality of the clause are provided,
    only phase 3 will be performed.

    The expected content of the input json are:
    * title: the title of the clause.
    * text: the textual content of the clause.
    * domain: the domaine the clause belong to (ex: ToS or service contracts).
    * type: type of the clause (optionnal)
    * anomalous: acceptabiity of the clause (optionnal)

    :return: Answer json including: title, type, anomalous and anomaly.
    :rtype: json
    """
    if not request.json or \
        not 'text' in request.json or \
        not 'title' in request.json or \
        not 'domain' in request.json or \
        not 'type' in request.json or \
        not 'anomalous' in request.json:
        logging.error("Post message wrongly formatted !")
        abort(400)
    else:
        logging.info("Dynamically classifiying and generating answer...")

        message = request.json

        cType = None
        isAnom = None
        aType = None

        # Nothing is provided: full analysis
        if message['type'] == "":
            logging.info("Performing complete analysis")
            cType, isAnom, aType = ClauseAnalyzer.analyzeClause(message['text'], message['title'])
        # Only type is provided
        elif message['anomalous'] == "":
            logging.info("Performing beta & gamma analysis")
            cType = message['type']
            isAnom, aType = ClauseAnalyzer.analyzeClauseBetaGamma(message['text'], message['type'])
        # Type and anomalous are provided
        else: 
            logging.info("Performing gamma analysis")
            cType = message['type']
            if message['anomalous'] == "True" or message['anomalous'] == 1:
                isAnom = True
                aType = ClauseAnalyzer.analyzeClauseGamma(message['text'], message['type'])
            else:
                isAnom = False
                aType = None

        answer = {
            'title': message['title'],
            'type': str(cType),
            'anomalous': str(isAnom),
            'anomaly': aType
        }
        return jsonify(answer), 201


@APP.route('/api/classsify_alpha', methods=['POST'])
def alphaClassifyText():
    """ alphaClassifyText

    Analyse the a given clause to determine its type.

    The expected content of the input json are:
    * title: the title of the clause.
    * text: the textual content of the clause.
    * domain: the domaine the clause belong to (ex: ToS or service contracts).

    :return: Answer json including: title, type.
    :rtype: json
    """
    if not request.json or \
        not 'text' in request.json or \
        not 'title' in request.json or \
        not 'domain' in request.json:
        logging.error("Post message wrongly formatted !")
        abort(400)
    else:
        logging.info("Alpha classifiying and generating answer...")

        message = request.json

        cType = ClauseAnalyzer.analyzeClauseAlpha(message['text'], message['title'])

        answer = {
            'title': message['title'],
            'type': str(cType)
        }
        return jsonify(answer), 201

if __name__ == '__main__':
    APP.run(debug=True, port=int("5555"))
