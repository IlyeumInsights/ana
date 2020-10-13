"""Module KbManager
*****************************

Manage an ontological Knowledge Base using owlready 2.
Apply rules (loaded from RulesManager) on KB.

Class
    KbManager
"""

import os
from owlready2 import *

from ana import Settings
from ana.knowledge.RulesManager import RulesManager, loadRawRules
from ana.utils import TextractTools

# Ontological structure

CONTRACT_KB = get_ontology("http://ilyeum.com/contract.owl")

with CONTRACT_KB:
    class Policy(Thing):
        pass
    class has_id(Policy >> str):
        pass
    class has_description(Policy >> str):
        pass

    class Text(Thing):
        """Text in a legal sense. Includes clauses, sentences, etc.
        """
        pass
    class has_text(Text >> str, FunctionalProperty):
        pass
    class has_subtext(Text >> Text):
        pass
    class has_position_in_parent(Text >> str):
        """Offset in number of char from the begging of parent text.
        """
        pass
    class has_duration(Text >> int): 
        """Duration of validity of the text (mainly the clause)
        """
        pass

    class Clause(Text):
        pass
    class has_title(Clause >> str, FunctionalProperty):
        pass
    class is_valid(Clause >> bool):
        pass
    class violates_policy(Clause >> Policy):
        pass

    class Sentence(Text):
        pass

    class Concept(Thing):
        pass
    class has_name(Concept >> str, FunctionalProperty):
        pass
    class has_terms(Concept >> str):
        pass

    class Type(Concept):
        pass
    class has_type(Clause >> Type, FunctionalProperty):
        """TODO
        """
        pass

    class Contract(Thing):
        pass
    class has_clause(Contract >> Clause):
        pass

class KbManager:
    """Class KbManager

    Set up and interact with the ontological knwoledge base.

    """
    def __init__(self):
        self.addConcepts()
        self.loadPolicyRules(Settings.KL_LOC+f"rules\\policies.swrl") # TODO
        self.loadOntoRules(Settings.KL_ONTO_RULES)

    def addRules(self, rules):
        with CONTRACT_KB:
            for rule in rules:
                Imp().set_as_rule(rule)


    def loadPolicyRules(self, path):
        """Relies on Rules Manager to load, format and add rules

        :param path: [description]
        :type path: [type]
        """
        rman = RulesManager()
        rules = rman.rules

        formatRules = []

        for policy in rules:
            Policy(has_id=policy)
            for rule in rules[policy]:
                ruleStr = ','.join(rule[0])
                ruleStr += ", Policy(?p), has_id(?p, '"+str(policy)+"')"
                ruleStr += " -> "+rule[1]
                ruleStr += ", violates_policy(?p)"
                formatRules.append(ruleStr)

        self.addRules(formatRules)

    def loadOntoRules(self, path):
        rules = loadRawRules(path)
        self.addRules(rules)

    def addConcepts(self):
        # Types
        typesDict = self.loadTypesTerms()
        for clType in typesDict:
            newType = Type(has_name=clType)
            newType.has_terms = typesDict[clType]

        # Concepts
        concDict = self.loadConceptTerms()
        for concept in concDict:
            newConcept = Concept(has_name=concept)
            newConcept.has_terms = concDict[concept]

    def loadConceptTerms(self):
        return self.loadTerms(Settings.KL_VOCAB_CONCEPT)

    def loadTypesTerms(self):
        return self.loadTerms(Settings.KL_VOCAB)

    def loadTerms(self, vocabPath):

        termDict = {}

        if os.path.isdir(vocabPath):
            for vocFile in os.listdir(vocabPath):
                conType = os.path.splitext(vocFile)[0]
                vocFilePath = os.path.join(vocabPath, vocFile)
                if os.path.isfile(vocFilePath):
                    # Access file that is composed of words separated my spaces
                    with open(vocFilePath, "r") as vocFileReader:
                        termDict[conType] = vocFileReader.read().split()

        return termDict


    def reason(self):
        sync_reasoner_pellet(
            infer_property_values = True,
            infer_data_property_values = True)


    def addAndAnalyzeClause(self, clauseTitle, clauseBody):
        clause = Clause(has_title=clauseTitle, has_text=clauseBody)

        # Subtext decompositions
        sentences = []
        for sentStr in clauseBody.split("."):
            
            newSent = Sentence(has_content=sentStr+".")
            
            # Perform sentence analysis
            d = TextractTools.extractDurationDay(sentStr)
            if d is not None:
                newSent.has_duration.append(d)

            sentences.append(newSent)

        clause.has_subtext = sentences

        # Apply reasoning
        self.reason()

        # Gather result
        print(clause.has_type)
        print(clause.is_valid)
        print(clause.violates_policy)


kbm = KbManager()
kbm.addConcepts()