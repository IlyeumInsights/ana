"""Module RulesManager
*****************************

Load rules as dictionnary and analyze them (it does no evaluate them for now).
Rule based evaluation, based on a KB, is a perspective.

Class
    RulesManager
"""
import re
import logging

from ana import Settings

class RulesManager:
    """class RulesManager

    Load a set of SWRL rules that represent policies, and analyzes them.
    Rules are stored in a file provided (path) to the constructor.

    :param rules: The set of rules reprensented as dictionnary that associates
        identifier (policy name) to an actual rule.
    :type rules: dict(str, str)
    """

    def __init__(self, rulesPath=Settings.KL_RULES):
        self.rules = {}
        self.loadRules(rulesPath)

    def loadRules(self, rulesPath):
        """loadRules

        Load a set of rules from a SWRL file.

        :param rulesPath: Path to the SWRL file.
        :type rulesPath: str (filePath)
        """
        self.rules = {}
        with open(rulesPath, "r") as rulesFile:
            for line in rulesFile.readlines():
                if len(line.strip()) <= 0 or line.strip().isspace() or line.strip()[0] == "#":
                    pass
                else:
                    key, val = line.split(":")
                    if key not in self.rules:
                        self.rules[key] = []
                    antecedent, consequent = val.split("->")
                    self.rules[key].append([antecedent.split("^"), consequent.strip()])



    def gatherClass(self):
        """gatherClass

        Gather the classes mentionned in the rules

        :return: The classes indentified in the rules.
        :rtype: list(str)
        """
        classes = []

        for polId in self.rules:
            for rule in self.rules[polId]:
                for predicate in rule[0]:
                    if "type" in predicate:
                        nclasses = re.findall('"([^"]*)"', predicate)
                        for nclass in nclasses:
                            if nclass not in classes:
                                classes.append(nclass)

        return classes

    def getPoliticsOfClass(self, clauseType):
        """getPoliticsOfClass

        Gather the policies related to one clause type according to the rules.

        :param clauseType: Type of clause.
        :type clauseType: str

        :return: Policies related to the given clause type
        :rtype: list(str)
        """
        politics = []

        for polId in self.rules:
            for rule in self.rules[polId]:
                for predicate in rule[0]:
                    if "type" in predicate:
                        nclasses = re.findall('"([^"]*)"', predicate)
                        for nclass in nclasses:
                            if clauseType == nclass and polId not in politics:
                                politics.append(polId)

        return politics

    def politicAppliesToSentence(self, politic):
        """politicAppliesToSentence

        Assert if a politic can be applied to a sentence or a complete clause.
        If true, classifier can be applied on sentences extraced from clause.
        To evaluate this assertion, rules predicate are reviewed.
        For now, predicate compatible with sentences are hard coded
        By default: False

        :param politic: Policy to check. (TODO rename)
        :type politic: str
        :return: True is the policy applies to a sentence, False if it applies
            to a full clause, defaults to False.
        :rtype: bool
        """
        # Anything that can be disciminized on sentence
        sentCompatibleList = ["greaterThan", "lessThan", "contains"]
        sentIncompatibleList = ["notContains", "hasSubject"]

        if politic not in self.rules:
            # TODO logging error
            return False

        res = None

        for rule in self.rules[politic]:
            for antecedent in rule[0]:
                # Get predicate name
                predicate = (antecedent.split("(")[0]).strip()
                if predicate in sentIncompatibleList:
                    res = False
                elif predicate in sentCompatibleList and res is None: # priority on false
                    res = True

        if res is not None:
            return res
        else:
            return False


if __name__ == "__main__":
    # rules = loadRules()
    # logging.debug(rules)
    # classes = gatherClass(rules)
    # logging.debug(classes)

    RM = RulesManager()
    logging.debug(RM.politicAppliesToSentence("R5"))
