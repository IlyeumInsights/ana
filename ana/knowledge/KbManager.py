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

class KbManager:
    """Class KbManager

    Set up and interact with the ontological knwoledge base.

    """
    def __init__(self):
        self.addConcepts()
        self.loadPolicyRules(Settings.KL_LOC+f"rules\\policies_owlready.swrl") # TODO
        self.loadOntoRules(Settings.KL_ONTO_RULES)

    def addRules(self, rules):
        with CONTRACT_KB:
            for rule in rules:
                print(rule)
                Imp().set_as_rule(rule)


    def loadPolicyRules(self, path):
        """Relies on Rules Manager to load, format and add rules

        :param path: [description]
        :type path: [type]
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


        print(clause.has_text)
        # Apply reasoning
        self.reason()

        # Gather result
        if clause.has_type:
            ctype = str(clause.has_type.has_name)
            print(ctype)
        else:
            ctype = None
        cVali = str(clause.is_valid)
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
        conceptInText = []
        conceptNotInText = []
        for concept in Concept.instances():
            cterms = concept.has_terms
            if any(term in text.lower() for term in cterms):
                conceptInText.append(concept)
            else:
                conceptNotInText.append(concept)
        return (conceptInText, conceptNotInText)




kbm = KbManager()

text = "le sous-traitant engage ne pas, directement ou indirectement, engager tout collaborateur agyla, meme si la sollicitation vient dudit collaborateur, en qualite de salarie ou de prestataire independant, le temps du present contrat et une annee apres son terme qu elle qu en soit la cause. defaut, et sans prejudice eventuels dommages et interets, le sous-traitant engage regler agyla en une fois, premiere demande, et sans delai, une somme representant douze fois la reemuneration mensuelle dudit collaborateur, calculee par la moyenne des remunerations mensuelles percues au titre des trois derniers mois de collaboration entre agyla et ledit collaborateur. le sous-traitant ne peut ceder le present contrat sans accord prealable et ecrit agyla."

print( kbm.addAndAnalyzeClause("Non solicitation", text) )
print( kbm.addAndAnalyzeClause("Non paiement", text) )
print( kbm.addAndAnalyzeClause("Non paiement", text) )
