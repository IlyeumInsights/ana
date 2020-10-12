"""Module DataCleaning
*****************************

Carries functions dedicated to data cleaning or text management.

:Authors: Nathan Ramoly
"""

import unidecode
import text_to_num

def normalizeText(text):
    """normalizeText

    Normalizes raw text by removing accents and special characters, lowering the
    text and removing trailing spaces.

    :param text: Text to normalize.
    :type text: str

    :return: Normalized text.
    :rtype: str
    """
    # Replace accents
    text = unidecode.unidecode(text)

    # Lowercase the text
    text = text.lower()

    # Removed whitespace (or char in param) at begining and end of string
    text = text.strip()

    return text

def mergeTextLines(text):
    """mergeTextLines

    Remove line breaks.

    :param text: Text with line breaks.
    :type text: str

    :return: Text without line breaks.
    :rtype: str
    """
    return text.replace("\n", " ")

def clean(text):
    """clean

    Performs the actual cleaning: normalize, merge lines and filter short words.

    :param text: Text to clean.
    :type text: str

    :return: Clean text.
    :rtype: str
    """
    text = normalizeText(text)
    text = mergeTextLines(text)
    text = filterText(text, 1)
    return text


def filterWords(tokens, threshold=3, numbers=True):
    """filterWords

    Remove short (size <= threshold) words that are irrelevant in analysis.
    Numbers be ignored by the filter.

    :param tokens: List of words.
    :type tokens: list(str)

    :param threshold: Words shorter than the threshold are to be filtered.
    :type threshold: int

    :param numbers: Assert if numbers should be filtered. (True if numbers are
        not filtered). True by default.
    :type numbers: int

    :return: Filtered list of words.
    :rtype: list(str)
    """
    filTokens = []
    for word in tokens:
        if word.isdigit() and numbers:
            filTokens.append(word)
        elif len(word) > threshold:
            filTokens.append(word)
    return filTokens

def filterText(text, threshold=3, numbers=True):
    """filterText

    Remove short (size <= threshold) words that are irrelevant in analysis.
    Numbers be ignored by the filter.
    Performs this task by decomposing the text as a list of words and calling
    'filterWords'.

    :param text: Text to filter.
    :type text: str

    :param threshold: Words shorter than the threshold are to be filtered.
    :type threshold: int

    :param numbers: Assert if numbers should be filtered. (True if numbers are
        not filtered). True by default.
    :type numbers: int

    :return: Filtered text.
    :rtype: str
    """
    return " ".join(filterWords(text.split(), threshold, numbers))

def convertNumber(text, lang="fr"):
    """convertNumber

    Converts all written numbers into figures.
    Relies on the text2num library.
    Example: "sixty cars" -> "60 cars"

    :param text: The text to convert
    :type text: str
    :param lang: Language of the text (en, fr or es)
    :type text: str
    """
    text = text_to_num.alpha2digit(text, lang)
