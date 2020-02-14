"""Module Election
************************

Election/voting procedure for selection and/or decision making according to
multiple sources.
Election can be configured in various way, and exported for further elections.

Classes:
    Candidate
        Represents a possible option in the election and its counts. It mainly
        carries comparison operators.
    Election
        Carries election modalities and performs the vote.

:Authors: Nathan Ramoly
"""

import logging

class Candidate:
    """class Candidate

    Represents a possible election options.
    This class carries comparison operators and its main role is to ease
    comparison between candidates.

    :param value: Name/id of the candidate
    :type value: str
    :param count: Number of vote for this candidate.
    :type count: int
    :param pcount: Ponderated count.
    :type pcount: int
    """

    def __init__(self, value, count=0, pcount=0):
        self.value = value # Name/id of the candidate
        self.count = count # True count
        self.pcount = pcount # Ponderated count (same as count by default)

    def __lt__(self, other):
        if self.pcount < other.pcount:
            return True
        elif self.pcount > other.pcount:
            return False
        elif self.count < other.count:
            return True
        else:
            return False

    def __gt__(self, other):
        if self.pcount > other.pcount:
            return True
        elif self.pcount < other.pcount:
            return False
        elif self.count > other.count:
            return True
        else:
            return False

    def __eq__(self, other):
        return self.pcount == other.pcount and self.count == other.count

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)

    def __str__(self):
        return str(self.value)+" (pc="+str(self.pcount)+", c="+str(self.count)+")"


class Election:
    """class Election

    Represents the election process for decision making or selection.
    Several parameters can be adjusted for the election.

    :param path: Path of the serialized election to import. 'None' by default.
    :type path: str (filepath)

    :param mode: Vote mode; defines the way ballots are counted.
        - 'std' (standard): normal counting
        - 'ato' (at least one): count the default value
        - 'pnd' (ponderated): count with weighted ballots. Without provided
        weights, the order defines the ponderation.
        - 'not' (nothing but): count by excluding the selected candidate.
    :type mode: str

    :param default: Default value, behavior varies according to the mode:
        - std:he default value is the selected one in case of equality.
        - ato: the default value is the one that is counted.
        - not: values different than the default value are returned.
        - pnd:the default value is the selected one in case of equality.
    :type default: str

    :param priority: Priority mode.
        - 'first': asceding importance (first ballots have more weights)
        - 'last': descending importance (last ballots have more weights)
    :type priority: str

    :param resultlvl: Verbosity of the output.
        - 'min' (minimal): returns only the candidate with the most votes and
        priority.
        - 'ele' (elected): returns all candidates with the most votes (equality)
        - 'ful' (full): returns all candidates in order associated with their
        respective counts.
    :type resultlvl: str

    """

    VOTEMODE = ["std", "ato", "pnd", "not"]

    PRIOMODE = ["first", "last"]

    RESLEVEL = ["min", "ele", "ful"]

    def __init__(self, path=None, mode=None, default=None, priority=None, resultlvl=None):
        """construction method

        Can initialize an Election object from parameter and from a serialized
        file.
        """
        if path is None:
            # Mode
            if mode is None or mode not in self.VOTEMODE:
                self.mode = self.VOTEMODE[0]
            else:
                self.mode = mode
            # Default
            self.default = default
            # Priority
            if priority is None or priority not in self.PRIOMODE:
                self.priority = None
            else:
                self.priority = priority
            # List
            if resultlvl is None or resultlvl not in self.RESLEVEL:
                self.resultlvl = "min"
            else:
                self.resultlvl = resultlvl
        else:
            # Import from file
            with open(path, 'r') as importFile:
                serial = importFile.read()
            params = serial.split("\n")

            if len(params) < 4:
                logging.error("Invalid election import file !")
                self.mode = self.VOTEMODE[0]
                self.default = default
                self.priority = None
                self.resultlvl = "min"
            else:
                # Mode
                if params[0] is None or len(params[0]) <= 0 or params[0] not in self.VOTEMODE:
                    self.mode = self.VOTEMODE[0]
                else:
                    self.mode = params[0]
                # Default
                if default is not None: # Default value has priority over saved one
                    self.default = default
                elif params[1] is None or len(params[1]) <= 0:
                    self.default = default
                else:
                    self.default = params[1]
                # Priority
                if params[2] is None or len(params[2]) <= 0 or params[2] not in self.PRIOMODE:
                    self.priority = None
                else:
                    self.priority = params[2]
                # List
                if params[3] is None or len(params[3]) <= 0 or params[3] not in self.RESLEVEL:
                    self.resultlvl = "min"
                else:
                    self.resultlvl = params[3]

    def counting(self, ballots, ponderation=None):
        """counting method

        Performs the vote and countings.

        :param ballots: List of elements that are to be counted. Elements should
            be comparable.
        :type ballots: list(any)

        :param ponderation: Associated list of weights. There should be as
            many weights as there are elements in the ballots list.
        :type ponderation: list(float)

        :return: Selected element(s) or list of elements with their count.
            (depends on the selected 'resultlvl')
        """
        logging.debug("Counting: "+str(ballots))
        if self.mode == self.VOTEMODE[0]:
            count = self._countingStd(ballots)
        elif self.mode == self.VOTEMODE[1]:
            count = self._countingAto(ballots)
        elif self.mode == self.VOTEMODE[2]:
            count = self._countingPnd(ballots, ponderation=ponderation)
        elif self.mode == self.VOTEMODE[3]:
            count = self._countingNot(ballots)
        else:
            logging.error("Counting mode "+str(self.mode)+" unknown !")
            return None

        # if count is None or len(count) <= 0:
        if count is None or not count:
            logging.warning("Counting 0 candidates !")
            return None

        # Odering of candidate
        sortedCandidates = []
        for c in count:
            insertPos = 0
            while insertPos < len(sortedCandidates):
                if c > sortedCandidates[insertPos]:
                    break
                elif c == sortedCandidates[insertPos]:
                    if c.value == self.default:
                        break
                    elif self.priority is not None and self.priority == self.PRIOMODE[1]:
                        break
                    else:
                        insertPos += 1
                else:
                    insertPos += 1
            sortedCandidates.insert(insertPos, c)

        logging.info("Vote result: ")
        for c in sortedCandidates:
            logging.info(c)

        if self.resultlvl == self.RESLEVEL[2]:
            return [(c.value, c.pcount) for c in sortedCandidates]
        elif self.resultlvl == self.RESLEVEL[1]: # only winners
            # if len(sortedCandidates) > 0:
            if sortedCandidates:
                ref = sortedCandidates[0].pcount # first pcount            
                return [(c.value, c.pcount) for c in sortedCandidates if c.pcount >= ref]
            else:
                return []
        else:
            return sortedCandidates[0].value

    def _countingStd(self, ballots):
        count = {}
        for c in ballots:
            if c not in count:
                count[c] = Candidate(c, 1, 1)
            else:
                count[c].count += 1
                count[c].pcount += 1

        return count.values()

    def _countingAto(self, ballots):  
        if self.default is None:
            logging.warning("No default value for ATO mode !") 
        if self.default in ballots:
            return [Candidate(self.default, ballots.count(self.default),
                              ballots.count(self.default))]
        else:
            return []

    def _countingPnd(self, ballots, ponderation=None):        
        if ponderation is None:
            if self.priority == self.PRIOMODE[0]:                
                ponderation = list( reversed( range(1, len(ballots)+1) ) )                
            else:
                ponderation = list( range(1, len(ballots)+1) )
        if len(ponderation) != len(ballots):
            logging.error("Ballots ("+str(len(ballots))+") and their ponderations ("+str(len(ponderation))+") are mismatching !")
            return None

        count = {}
        pos = 0
        for c in ballots:            
            if c not in count:
                count[c] = Candidate(c, 1, ponderation[pos])
            else:
                count[c].count += 1
                count[c].pcount += ponderation[pos]
            pos += 1

        return count.values()

    def _countingNot(self, ballots):
        count = {}
        countDef = 0 # Count of the value to exclude

        if self.default is None:
            logging.warning("Value to exclude is not set ! ('default' parameter)")

        for c in ballots:
            if c == self.default:
                countDef += 1
            else:
                if c not in count:
                    count[c] = Candidate(c, 1, 1)
                else:
                    count[c].count += 1
                    count[c].pcount += 1
        if len(count) > 0:
            return count.values()
        else:
            # outdict = {}
            # outdict[c] = Candidate(self.default, countDef, countDef)
            return [Candidate(self.default, countDef, countDef)]        

    def export(self, path):
        """export method

        Exports election modalities (parameters) into a file.
        It can be seen as an election template.

        :param path: Export destination.
        :type path: str (filePath)
        """
        if path is None:
            logging.error("Election export path is not set !")
            return

        serial = ""
        serial += str(self.mode)+"\n"
        serial += str(self.default)+"\n"
        serial += str(self.priority)+"\n"
        serial += str(self.resultlvl)+"\n"

        with open(path, "w") as exportFile:
            exportFile.write(serial)


if __name__ == "__main__":
    # Debug
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.DEBUG)

    VOTE = Election(resultlvl="min", priority="last", mode="not", default=0)

    BALLOTS = [0, 1, 2, 0, 0, 1]

    print(VOTE.counting(BALLOTS))
