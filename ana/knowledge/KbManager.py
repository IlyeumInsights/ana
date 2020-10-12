"""Module KbManager
*****************************

Manage an ontological Knowledge Base using owlready 2.
Apply rules (loaded from RulesManager) on KB.

Class
    KbManager
"""

from owlready2 import *

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
        self.loadConceptTerms()

    def addRules(self, rules):
        pass

    def loadRules(self, path):
        """Relies on Rules Manager to load and add rules

        :param path: [description]
        :type path: [type]
        """
        pass

    def loadConceptTerms(self):
        pass

    def reason(self):
        pass

    def addAndAnalyzeClause(self, clauseTitle, clauseBody):
        pass