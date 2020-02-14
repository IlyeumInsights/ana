"""Module DataAcquisition
**************************
Acquires annotated texts (raw or json), stores them in a db and generates
dataset for the learning phase.
WIP

:Authors: Nathan Ramoly
"""
import uuid
import logging
import datetime

from ana.knowledge import DbManager


def acquireJSONContract(domain, jsoncontract):
    """acquireJSONContract
    
    Acquires an instance (version) of a contract as a JSON, stores in in db
    according to matching domain (ilyeum, eula, etc.) and parent contract
    (a contract have serveral instances/versions during negotiation).

    TODO Check previous contract similarity
    """
    DbManager.storeDocInstance(domain, jsoncontract)


def createNewDocument(name, description, author, client):
    """
    Create a new document/contract as a json format
    Format:

    .. code-block:: guess

        {
            _id: uuid
            name: string
            description: string
            author: string
            client: string
        }
    """
    return {"_id":uuid.uuid4(),"name":name, "decription":description, "author":author, "client":client}

def acquireNewContract(domain, jsonContractInfo):
    DbManager.createTextDocument(domain, jsonContractInfo)


def exportDataset(domain, levels):
    pass

class DocumentAcquisitor():
    """class DocumentAcquisitor

    Incrementaly create a JSON annotated documents instance by going through
    subparts (clause, sentence).
    Created JSON structure (wip):

    .. code-block:: guess

        {
            parentDocument: uuid
            version: version of the current instance of contracts (?)
            date: date of the upload or insertion (if not provided)
            content:
            {
                level1:
                [{
                    _id:
                    level: ?
                    parentText: id
                    position: position in parent text (offset)
                    text: string
                    annotations:
                    {
                        dimension1: annotation1
                        ...
                        dimensionN: annotationN
                    }
                }]
                level2:
                [{
                    ...
                }]
            }
        }
    """
    jsonDocInstance = {}


    def __init__(self, domain, name, description=None, version=None, date=datetime.datetime.now()):
        # TODO check if contract exists, if it does , select name from it,
        # otherwise, create it

        self.domain = domain
        self.jsonDocInstance = {}
        self.jsonDocInstance["parentDocument"] = None # TODO
        self.jsonDocInstance["version"] = version # TODO (from prev vers)
        self.jsonDocInstance["date"] = date
        self.jsonDocInstance["content"] = {}

    def addAnnotatedSubtext(self, text, level, annotation, id=uuid.uuid4(), parentText=None, position=None):
        """
        id is an uuid or a given string ("clause1")
        Annotation is a dict {dimensions:annotations} (ex {type:pay, anom:R4})
        """
        subtext = {}
        subtext["_id"] = id
        subtext["parentText"] = parentText
        subtext["level"] = level # duplicated with upper level dict
        subtext["position"] = position
        subtext["text"] = text
        subtext["annotations"] = annotation

        # Creating level (clause, sentence...) if doesn't exist
        if level not in self.jsonDocInstance["content"]:
            self.jsonDocInstance["content"][level] = []
        self.jsonDocInstance["content"][level].append(subtext)
        
        pass

    def export(self):
        return self.jsonDocInstance

    def debugDisplay(self):
        logging.debug(str(self.jsonDocInstance))

    def store(self):
        acquireJSONContract(self.domain, self.jsonDocInstance)


if __name__ == "__main__":

    da = DocumentAcquisitor("ilyeum", "contrat1")
    da.addAnnotatedSubtext("Ceci est un texte d'essai. Avec deux phrases", "clause", {"type":"test"}, id="clause1")
    da.addAnnotatedSubtext("Ceci est un texte d'essai.", "sentence", {"type":"test"}, parentText="clause1")
    da.addAnnotatedSubtext("Avec deux phrases.", "sentence", {"type":"fin"}, parentText="clause1")

    da.debugDisplay()

    da.store()
