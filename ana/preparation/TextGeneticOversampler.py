"""Module TextGeneticOversampler
*****************************

Provide a class that enrich a text dataset by exchanging parts between two
texts, as perform in chromosomes before mitosis.

Class
    TextGeneticOversampler
"""

import random
import math
import logging
import numpy


class TextGeneticOversampler:
    """class TextGeneticOversampler

    Oversamples text samples by generating new sentences by genetic-like merge.

    :param generation: Number of exchange process to apply.
    :type generation: int
    """

    generation = 1

    def __init__(self, generation=1):
        self.generation = generation


    def packText(self, text, nbPack):
        """packText

        Transforms a text as a set of pack of words.
        By default, create packs containing one word each.
        Packs the granule that are used to create further sentences.

        :param text: Text to decompose as a set of pack.
        :type text: str
        :param nbPack: Number of expected packs.
        :type nbPack: int
        :return: A set of pack of words.
        :rtype: list(list(str))
        """        
        if nbPack == 0:
            return text.split(" ")
        else:
            split = text.split(" ")
            wordPerPack = int(len(split)/nbPack)
            packs = [] # List of packs
            curPack = []
            for i in range(0, nbPack):
                if i%wordPerPack == 0:
                    packs.append(curPack)
                    curPack = []
                curPack.append(split[i])
            # Adding last
            if len(curPack) > 0:
                packs.append(curPack)
            return packs

    def geneticTextMerge(self, textA, textB, ordered=False, nbPack=0, mutaRate=0.25):
        """geneticTextMerge

        Exchanges packs (subtext) of two sentences/texts, thus generating
        variation of both texts.

        :param textA: First text.
        :type textA: str
        :param textB: Second text.
        :type textB: str
        :param ordered: assert if pack/word should be kept in initial order,
            defaults to False
        :type ordered: bool, optional
        :param nbPack:  number of text parts that will be exchanged,
            defaults to 0
        :type nbPack: int, optional
        :param mutaRate: Mutation rate that represents the rate of exchanged
            parts in the texts, defaults to 0.25
        :type mutaRate: float, optional
        :return: First mutated text.
        :rtype: str
        :return: Second mutated text.
        :rtype: str
        """

        packsA = self.packText(textA, nbPack)
        packsB = self.packText(textB, nbPack)

        # A is the smallest set
        if len(packsA) > len(packsB):
            packsBuffer = packsB
            packsB = packsA
            packsA = packsBuffer

        newPackA = []
        newPackB = []
        randPackB = packsB.copy()
        for i in range(0, len(packsA)):
            if random.random() < mutaRate:
                if ordered:
                    # Exchange with ith element in B
                    newPackA.append(packsB[i])
                    newPackB.append(packsA[i])
                else:
                    # Select a random element in B
                    # Caution: a same element can be selected multiple times
                    randPos = math.floor(random.random()*len(randPackB))
                    newPackA.append(randPackB[randPos])
                    newPackB.append(packsA[i])
                    del randPackB[randPos]
            else:
                if ordered:
                    newPackA.append(packsA[i])
                    newPackB.append(packsB[i])
                else:
                    newPackA.append(packsA[i])
                    randPos = math.floor(random.random()*len(randPackB))
                    newPackB.append(randPackB[randPos])
                    del randPackB[randPos]

        # Complete with the rest
        if ordered:
            for i in range(len(packsA), len(packsB)):
                if random.random() < mutaRate:
                    newPackA.append(packsB[i])
                else:
                    newPackB.append(packsB[i])
        else:
            for randRemain in randPackB:
                if random.random() < mutaRate:
                    newPackA.append(randRemain)
                else:
                    newPackB.append(randRemain)

        return " ".join(newPackA), " ".join(newPackB)


    def fit_resample(self, X, Y):
        """fit_resample

        Performs the actual overfitting by generating new samples/sentences by
        by genetic combination per class.
        
        :param X: Text samples.
        :type X: np.array
        :param Y: Annotation for each text of X
        :type Y: np.array
        :return: Oversampled text samples set.
        :rtype: np.array
        :return: Oversampled text annotation.
        :rtype: np.array
        """

        # Generation loop
        for g in range(0, self.generation):

            newX = []
            newY = []

            unique, counts = numpy.unique(Y, return_counts=True)

            # Compute ratio for generation
            maxCount = 0
            for count in counts:
                if count > maxCount:
                    maxCount = count

            ratio = {}
            for i in range(0, len(unique)):
                ratio[unique[i]] = float(maxCount)/float(counts[i])

            # The rario provide the number of sentence per... sentence

            # Go through all sentences and generate sentences
            for i in range(0, len(X)):
                countGenSent = 0

                # Add orig
                newX.append(X[i])
                newY.append(Y[i])

                # TODO random order
                for j in range(i+1, len(X)):

                    labelA = Y[i]
                    labelB = Y[j]

                    if labelA == labelB:

                        countGenSent += 2

                        sentA, sentB = self.geneticTextMerge(X[i], X[j])

                        # new ones
                        newX.append(sentA)
                        newY.append(Y[i])
                        newX.append(sentB)
                        newY.append(Y[i])

                        # Break if we created enough sentence
                        if countGenSent > ratio[labelA]:
                            break
                # End sub sentence loop
            # End main sentence loop

            X = numpy.array(newX)
            Y = numpy.array(newY)
        # End Generation loop

        return X, Y


# Test
if __name__ == '__main__':
    sentA = "Ceci est une phrase très jolie."
    sentB = "Celle là ne l'est pas, non mais voyons donc."
    sentC = "Une phrase d'un autre genre, sans foi, ni loi"
    sentD = "Pareil pour cette phrase qui n'est sous aucune autorité"
    sentE = "Cette phrase entre deux, jolie mais pas trop"
    X = numpy.array([sentA, sentB, sentC, sentD, sentE])
    Y = numpy.array([1, 1, 0, 0, 1])
    tgo = TextGeneticOversampler()
    logging.debug(tgo.fit_resample(X, Y))
    # logging.debug(tgo.geneticTextMerge(sentA, sentB, ordered=True))