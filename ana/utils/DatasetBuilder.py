"""Module DatasetBuilder
*************************
Aggregate and transform data of annotated contracts.
Deprecated.
"""
import os
import shutil
import zipfile
import csv
import codecs


def aggregateData(pathDir):
    """
    Go through directories that each matchs a document and contains a dataset
    zip file.
    The content of the zip file is extracted and aggregated in a single dir.
    """

    # Number of analyzed documents
    docnum = 0

    # Output dir
    outputDir = os.path.join(pathDir, "..\\aggDataset\\") 
    outputTxt = os.path.join(outputDir, "texts\\") 
    outputCsv = os.path.join(outputDir, "classes.csv") 
    # Create dir
    os.mkdir(outputDir)
    os.mkdir(outputTxt)
    open(outputCsv, 'w+')

    for subdir, dirs, files in os.walk(pathDir):
        dspath = os.path.join(subdir, "datagen.zip")
        if os.path.isfile(dspath):
            # Extract Temporary dir
            outpath = os.path.join(subdir, "datagen\\")                     
            with zipfile.ZipFile(dspath,"r") as zip_ref:                
                zip_ref.extractall(outpath)

            # Define dir ID
            docnum += 1
            id = str(docnum).zfill(5)

            textdir = os.path.join(outpath, "texts")
            for f in os.listdir(textdir):
                shutil.copyfile(os.path.join(textdir, f), outputTxt+id+"_"+f)
            
            csvStrToAdd = ""
            delim = ","
            with open(os.path.join(outpath, "classes.csv"), 'r') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=delim)
                for row in csv_reader:
                    if row[0] != "id":
                        csvStrToAdd += id+"_"+row[0]+delim+row[1]+"\n"
            
            with open(outputCsv, "a") as outcsv:
                outcsv.write(csvStrToAdd)

            # Remove dir
            shutil.rmtree(outpath)


def extractSubDataset(datasetPath, firstLineOnly=False):
    """
    Extract all clauses from a given type (vocabulary)
    Work from extracted data set
    """
    # voc = ["résiliation", "resiliation" ]
    voc = ["financières", "financieres", "financière", "financiere", "financiéres", "financiére", "paiement", "facturation" ]

    delim = ","

    labels = {}

    outputDir = os.path.join(datasetPath, "..\\aggSubDataset\\") 
    outputTxt = os.path.join(outputDir, "texts\\") 
    outputCsv = os.path.join(outputDir, "classes.csv") 
    os.mkdir(outputDir)
    os.mkdir(outputTxt)
    open(outputCsv, 'w+')

    with open(os.path.join(datasetPath, "classes.csv"), 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=delim)
        for row in csv_reader:
                if row[0] != "id":
                    labels[row[0]] = row[1]

    # Go through text files
    csvStrToAdd = ""
    textdir = os.path.join(datasetPath, "texts")
    for f in os.listdir(textdir):
        with open(os.path.join(textdir, f), "r", encoding = 'utf8') as segFile:
            segStr = segFile.read()

            if firstLineOnly:
                segStr = segStr.split("\n")[0]
            
            print(f+" "+segStr.lower())

            if any(x.lower() in segStr.lower() for x in voc):
                # Move File
                shutil.copyfile(os.path.join(textdir, f), outputTxt+f)
                # Add line in csv
                id = f[:-4]
                csvStrToAdd += id+delim+labels[id]+"\n"

    with open(outputCsv, "a") as outcsv:
        outcsv.write(csvStrToAdd)


def convertFilesToUTF8(directory):
    """
    Convert all file of a directory from ANSI to UTF-8 (if not UTF-8)
    """
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            # UTF-8, we are good
            open(filepath, encoding='UTF-8').read() 
            print("File "+filename+" already encoded in UTF-8")
        except:

            print("Transforming file "+filename+" to UTF-8")
            filedata = open(filepath, encoding='ANSI').read() 
            # print(filedata)
            with codecs.open(filepath, 'w', encoding = 'utf8') as f:
                f.write(filedata)



# aggregateData("C:\\Users\\Nara\\Workspace\\datasets\\Contrats\\Raw\\")

# convertFilesToUTF8("C:\\Users\\Nara\\Workspace\\datasets\\Contrats\\Raw\\Audensiel1\\datagen\\texts\\")

# extractSubDataset("C:\\Users\\Nara\\Workspace\\datasets\\Contrats\\aggDataset\\", True)