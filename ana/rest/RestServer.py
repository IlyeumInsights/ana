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

    Classifies a text (supposidely a clause) provided by HTTP post in JSON.
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


if __name__ == '__main__':
    APP.run(debug=True, port=int("5555"))
