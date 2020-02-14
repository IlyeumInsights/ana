"""
Web server based on Flask.
Enable sentences and clauses annotation
@Author Nathan Ramoly - Ilyeum Lab
"""

from flask import Flask
from flask import render_template, request, flash, redirect, url_for, send_from_directory

import os
import shutil
import zipfile
import codecs
import csv
import logging




app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp'
app.secret_key = "svwxvb12hlpy1561sd216fh"

CLASSESPATHCLAU = "Annotation\\classes\\clause"
CLASSESPATHSENT = "Annotation\\classes\\sentence"
DATASETPATH =  "Annotation\\dataset"



def convertFilesToUTF8(directory):
    """
    Convert all file of a directory from ANSI to UTF-8 (if not UTF-8)
    """
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            # UTF-8, we are good
            open(filepath, encoding='UTF-8').read() 
            logging.info("File "+filename+" already encoded in UTF-8")
        except:

            logging.info("Transforming file "+filename+" to UTF-8")
            filedata = open(filepath, encoding='ANSI').read() 
            with codecs.open(filepath, 'w', encoding = 'utf8') as f:
                f.write(filedata)



def loadDataList(dirpath = "C:\\Users\\Nara\\Workspace\\datasets\\Contrats\\Raw"):
    """
    Go through directories (contract) and list each clauses
    Each list entry is a sublist as:
    0- client contract name
    1- path to contract/anomalies/datagen arch
    2- path to the clause
    """

    datalist = []

    for contract in os.listdir(dirpath):
        contpath = os.path.join(dirpath, contract)
        # Open the data gen zip
        datagenpath = os.path.join(contpath, "datagen.zip")
        outpath = os.path.join(contpath, "datagen\\")  

        #  if os.path.isfile(datagenpath): #Datagen as zip
        if os.path.isdir(outpath): # datagen already exists
            # Extract Temporary dir                   
            # with zipfile.ZipFile(datagenpath,"r") as zip_ref:                
            #     zip_ref.extractall(outpath)

            clausespath = os.path.join(outpath, "texts\\")

            # Transform dir content to utf-8
            convertFilesToUTF8(clausespath)

            # List all Clause            
            for clause in os.listdir(clausespath):
                clausepath = os.path.join(clausespath, clause)
                # Add list entry
                elem = []
                elem.append(contract) # 1- client contract name
                elem.append(contpath) # 2- path (leading to anomalies & prev annot)
                elem.append(clausepath) # 3- path of the clause
                datalist.append(elem)

            # Remove dir
            # shutil.rmtree(outpath)

    return datalist


def generateDataset(request):
    logging.info("Generating dataset... ")
    # logging.debug(request.form)
    
    # 1- Clause property and annotation
    # Loading clauses classes classes (no error in that comment)
    files = os.listdir(CLASSESPATHCLAU)
    clauseClasses = []
    for f in files:
        # Remove extension
        clauseClasses.append(os.path.splitext(f)[0])

    idClause = request.form["idclause"].zfill(4)

    csvClauseProp = idClause+", "+request.form["contName"]+", "+request.form["clTitle"].replace(",", " ")
    csvClauseAnnot = idClause
    for clClass in clauseClasses:
        csvClauseAnnot += ", "+request.form['clause_'+clClass]

    # logging.debug( csvClauseProp )
    # logging.debug( csvClauseAnnot )

    # TODO check and remove previous entry (for now, files should be emptied beforehand)
    with open(os.path.join(DATASETPATH, "clauseProp.csv"), "a") as clauseClassesFile:
        clauseClassesFile.write(csvClauseProp+"\n")
    
    with open(os.path.join(DATASETPATH, "clauseClasses.csv"), "a") as clauseClassesFile:
        clauseClassesFile.write(csvClauseAnnot+"\n")

    # 2- Sentences copy and annotation

    # Load sentence classes classes
    files = os.listdir(CLASSESPATHSENT)
    sentClasses = []
    for f in files:
        # Remove extension
        sentClasses.append(os.path.splitext(f)[0])
    logging.debug(sentClasses)
    
    nbElem = int(request.form['nbElem'])

    for i in range(1, nbElem+1):
        # If suppressed, ignore:
        if not request.form.get("sup_"+str(i)):
            # Get annotation
            # Gather Id and annotation
            idSent = idClause+"_"+str(i).zfill(3)

            # Get annotation
            csvSentAnnot = idSent
            for seClass in sentClasses:
                logging.debug('sent_'+str(i)+"_"+seClass)
                csvSentAnnot += ", "+request.form['sent_'+str(i)+"_"+seClass]

            # fill csv file
            with open(os.path.join(DATASETPATH, "sentClasses.csv"), "a") as sentClassesFile:
                sentClassesFile.write(csvSentAnnot+"\n")

            # Generate text files
            with open(os.path.join(DATASETPATH, "texts\\"+idSent+".txt"), "w", encoding='utf-8') as text_file:
                text_file.write( request.form.get("text_"+str(i)) ) 



@app.route('/', methods=['POST', 'GET'])
def index():
    """
    Main
    """
    # handling annotation
    if request.method == 'POST':
        generateDataset(request)

    # Next Annotation
    pos = 0
    if "pos" in request.args:
        pos = int(request.args["pos"])

    # Load clause at position pos in list
    contract = CLAUSELIST[pos][0]
    path = CLAUSELIST[pos][1]
    clausePath = CLAUSELIST[pos][2]

    # Load remarks
    anomFilePath = os.path.join(path, "anomalies.txt")
    remarks = "No remarks available"
    if os.path.isfile(anomFilePath):
        with open(anomFilePath, "r") as anomFile:
            remarks = anomFile.read()

    # Load previous classification
    delim = ","
    prevAnnotFile = os.path.join(path, "datagen\\classes.csv")
    prevAnnot = "NA"
    with open(prevAnnotFile, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=delim)
        for row in csv_reader:
            if row[0] == os.path.splitext( os.path.basename(CLAUSELIST[pos][2]) )[0]:
                prevAnnot = row[1]

    # Load raw clause
    logging.debug(clausePath)
    if os.path.isfile(clausePath):
        with open(clausePath, "r", encoding="utf8") as clauseFile:
            clause = clauseFile.read()

    clauseorig = clause
    # Decompose raw clause
    title = clause.split("\n")[0]
    clause = clause.replace("\n", " ")
    valList = clause.split(".")
    # Remove title from first elem
    valList[0] = valList[0].replace(title, "")
    # Re-add delimiter and strip
    for i in range(0, len(valList)-1):
        valList[i] += "."
        valList[i] = valList[i].strip()

    # Filter list
    filVaList = []
    for text in valList:
        if not text.isspace():
            filVaList.append(text)
    valList = filVaList # Too lazy to update list name after

    # Load classes
    files = os.listdir(CLASSESPATHCLAU)
    clauseClasses = {}
    for f in files:
        # Remove extension
        classCat = os.path.splitext(f)[0]
        with open(os.path.join(CLASSESPATHCLAU, f)) as classFile:
            clauseClasses[classCat] = classFile.read().split("\n")

    files = os.listdir(CLASSESPATHSENT)
    sentClasses = {}
    for f in files:
        # Remove extension
        classCat = os.path.splitext(f)[0]
        with  open(os.path.join(CLASSESPATHSENT, f)) as classFile:
            sentClasses[classCat] = classFile.read().split("\n")


    return render_template("gui.html", idclause=pos, nextpos=str(pos+1), valList=valList, contract=contract, remarks=remarks, rawclause=clauseorig, title=title, clauseClasses=clauseClasses, sentClasses=sentClasses, prevClAnnot = prevAnnot)


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

    CLAUSELIST = loadDataList("C:\\Users\\Nara\\Workspace\\datasets\\Contrats\\Raw2.0")
    logging.debug(CLAUSELIST)
    app.run()