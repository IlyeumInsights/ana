"""Module DbManager
*******************

Module in charges of storing the acquired annotated data in a database.
For now: MongoDB
WIP
"""
import logging
import pymongo

MGDB = 'mongodb://localhost:27017/'

def addDocumentInCollection(dbName, collection, jsonContent):
    """addDocumentInCollection

    Adds a document/contract (in JSON) in a collection in a database in MongoDB.

    :param dbName: Name of the database  to use in MongoDB.
    :type dbName: str
    :param collection: Collection to add the content.
    :type collection: str
    :param jsonContent: Content to insert in the db.
    :type jsonContent: json
    """
    client = pymongo.MongoClient(MGDB)

    # Opening/creating a db for the domain
    dba = client[dbName]

    # Adding entry collection
    textColl = dba[collection]

    iid = textColl.insert_one(jsonContent).inserted_id

    logging.info("Added element in collection %s in DB %s  with id %s ",
                 collection, dbName, str(iid))

    client.close()


def storeDocInstance(domain, jsonContractContent):
    """storeDocInstance

    Stores an instance (version) of a document in the databse.

    :param domain: Name of the the domain considered, such as Ilyeum contract,
        EULA, etc., this domain matches a dedicated db in MongoDB?
    :type domain: str
    :param jsonContractContent: Content to insert in the db.
    :type jsonContractContent: json
    """
    addDocumentInCollection(domain, "Texts", jsonContractContent)


def createTextDocument(domain, jsonContractInfo):
    """storeDocInstance

    Create a new document/contrac.
    Stores meta-information about a contract that are independent from
    its changing contents.

    :param domain: Name of the the domain considered, such as Ilyeum contract,
        EULA, etc., this domain matches a dedicated db in MongoDB?
    :type domain: str
    :param jsonContractInfo: Information about the document to insert in the db.
    :type jsonContractInfo: json
    """
    addDocumentInCollection(domain, "Documents", jsonContractInfo)
    