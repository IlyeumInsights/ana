"""Module KbManager
*****************************

Manage an ontological Knowledge Base using owlready 2.
Apply rules (loaded from RulesManager) on KB.

Note: this module follows a different naming style for ontological concepts.

Class
    KbManager
"""

import os
from owlready2 import *

from ana import Settings
from ana.knowledge.RulesManager import RulesManager, loadRawRules
from ana.utils import TextractTools
from ana.utils.Singleton import Singleton

# Ontological structure

CONTRACT_KB = get_ontology("http://ilyeum.com/contract.owl")

with CONTRACT_KB:
    class Policy(Thing):
        pass
    class has_id(Policy >> str, FunctionalProperty):
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
    class is_valid(Clause >> bool, FunctionalProperty):
        pass
    class violates_policy(Text >> Policy):
        pass
    class respects_policy(Text >> Policy):
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
    class has_concept(Text >> Concept):
        pass
    class has_not_concept(Text >> Concept):
        pass

    class Contract(Thing):
        pass
    class has_clause(Contract >> Clause):
        pass

class KbManager(metaclass=Singleton):
    """Class KbManager

    Set up and interact with the ontological knwoledge base.
    Singleton class as loading is to be performed once.

    """
    def __init__(self):
        """Constructor

        Ensure concepts and rules are loaded before adding any entities.
        """
        self.addConcepts()
        self.loadPolicyRules(Settings.KL_LOC+f"rules/policies_owlready.swrl") # TODO
        self.loadOntoRules(Settings.KL_ONTO_RULES)

    def addRules(self, rules):
        """addRules

        Add a set of rules in the ontology

        :param rules: SWRL ready and OwlReady2 rules 
        :type rules: [str]
        """
        with CONTRACT_KB:
            for rule in rules:
                # print(rule)
                Imp().set_as_rule(rule)


    def loadPolicyRules(self, path):
        """loadPolicyRules

        Relies on Rules Manager to load, format and add rules.

        :see: addRules

        :param path: Path of the rule files
        :type path: str
        """
        rman = RulesManager(path)
        rules = rman.rules

        formatRules = []

        for policy in rules:
            Policy(has_id=policy)
            for rule in rules[policy]:
                ruleStr = ','.join(rule[0])
                ruleStr += ", Policy(?p), has_id(?p, '"+str(policy)+"')"
                ruleStr += " -> "+rule[1]
                # ruleStr += ", violates_policy(?p)"
                formatRules.append(ruleStr)

        self.addRules(formatRules)

    def loadOntoRules(self, path):
        """loadOntoRules

        Load rules specific to the ontologies and not related to policies 
        validation rules.

        :param path: Path to the ontology rules fule
        :type path: str
        """
        rules = loadRawRules(path)
        self.addRules(rules)

    def addConcepts(self):
        """addConcepts

        Load concepts and add them in the ontology
        """
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
        """loadConceptTerms

        Load of the terms known for a concept from file.

        :return: Dictionnary of concept associated to a list of term.
        :rtype: {str, [str]}
        """
        return self.loadTerms(Settings.KL_VOCAB_CONCEPT)

    def loadTypesTerms(self):
        """loadConceptTerms

        Load of the terms known for a clause type (is a concept) from file.

        :return: Dictionnary of clause types associated to a list of term.
        :rtype: {str, [str]}
        """
        return self.loadTerms(Settings.KL_VOCAB)

    def loadTerms(self, vocabPath):
        """loadTerms

        Load terms for a concept from file.
        A file carries a string of terms separated by a space.

        :param vocabPath: Path to the directory of term
        :type vocabPath: str
        :return:  Dictionnary of concept associated to a list of term.
        :rtype: {str, [str]}
        """

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
        """reason

        Apply pellet reasonner.        
        Pellet doesn't like coma and quote !!
        """
        sync_reasoner_pellet(
            infer_property_values = True,
            infer_data_property_values = True)


    def addAndAnalyzeClause(self, clauseTitle, clauseBody):
        """addAndAnalyzeClause

        Add a single clause to the ontology and provide its key properties
        (validity, type, violated clause) after reasoning.
        Commas are removed from text content.

        :param clauseTitle: Title of the clause
        :type clauseTitle: str
        :param clauseBody: Content of the clause
        :type clauseBody: str
        :return: clause type, clause validaty as stringify boolean, list of
            violated policies, list of notable sentences
        :rtype: str, str, [str], [str]
        """
        clauseBody = clauseBody.replace(",", "")
        # print(clauseBody)

        clause = Clause(has_title=clauseTitle, has_text=clauseBody)

        # Clause analysis
        embConcept, nembConcept = self.checkConceptInText(clauseBody)
        clause.has_concept = embConcept
        clause.has_not_concept = nembConcept

        # Subtext decompositions
        sentences = []
        for sentStr in clauseBody.split("."):
            
            newSent = Sentence(has_text=sentStr+".")
            
            # Perform sentence analysis
            d = TextractTools.extractDurationDay(sentStr)
            if d is not None:
                newSent.has_duration = d

            embConcept, nembConcept = self.checkConceptInText(sentStr)
            newSent.has_concept = embConcept
            newSent.has_not_concept = nembConcept

            sentences.append(newSent)

        clause.has_subtext = sentences

        # Apply reasoning
        self.reason()

        # Gather result
        if clause.has_type:
            ctype = str(clause.has_type.has_name)
        else:
            ctype = None
        # cVali = str(clause.is_valid[0])
        if clause.is_valid is None:  # if None -> is valid is true
            cVali = True
        else: # Rules only set is_valid to false
            cVali = False
        cVPol = []
        if clause.violates_policy:
            for violatedPol in clause.violates_policy:
                cVPol.append(str(violatedPol.has_id))
        sentences = []
        if clause.has_subtext:
            for subtext in clause.has_subtext:
                if subtext.violates_policy:
                    sentences.append(subtext.has_text)

        return ctype, cVali, cVPol, sentences


    def checkConceptInText(self, text):
        """checkConceptInText

        List concepts of ontologies that are and are not in the input text.
        Performed as function to overcome limit of SWRL towards negations.

        :param text: Text to review to find concepts
        :type text: str
        :return: Lists of found concepts and not found concepts
        :rtype: [Concept], [Concept]
        """
        conceptInText = []
        conceptNotInText = []
        for concept in Concept.instances():
            cterms = concept.has_terms
            if any(term in text.lower() for term in cterms):
                conceptInText.append(concept)
            else:
                conceptNotInText.append(concept)
        return (conceptInText, conceptNotInText)

    def clearOntology(self):
        """clearOntology

        Clear all instances of ontology.
        Useful for performance issue when reasoning on many clauses that are not
        meant to be stored.
        """
        for textI in Text.instances():
            destroy_entity(textI)
