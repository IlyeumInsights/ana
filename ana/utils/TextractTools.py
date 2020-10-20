"""Module TextractTools

Contains various functions around data extraction from text.

"""
import re

# TODO gather from file or online resources
TERMSDAY = ["jour", "day", "journée"]
TERMSMON = ["mois", "month"]
TERMSYEA = ["an", "année", "year"]

def extractValue(text):
    """extract_value

    Find all numerical values with their units from the text.

    :param text: Text to extract numerical value from
    :type text: str
    :return: List of couple (value, unit)
    """
    p = re.compile(r"\d+\s+(?:\w+)")
    res = p.findall(text)

    retList = []
    for match in res:
        value, unit = match.strip().split(" ")
        value = int(value)
        retList.append((value, unit))

    return retList

def extractDurationDay(text):
    """extractDurationDay

    Find all duration mentionned in text and return their value per day as int.
    Example: "paiment performed under 1 month, i have 3 puppies"
    returns: 30 (days)

    :param text: Text to extract durations value from
    :type text: str
    :return: List of durations found in text in days.
    :rtype: [int]
    """

    numbers = extractValue(text)

    durationList = []

    for num in numbers:
        value, unit = num

        # unit[:-1] is done to match plural
        if unit in TERMSDAY or unit[:-1] in TERMSDAY:
            durationList.append(value)
        elif unit in TERMSMON or unit[:-1] in TERMSMON:
            durationList.append(value*30)
        elif unit in TERMSYEA or unit[:-1] in TERMSYEA:
            durationList.append(value*365)
        else:
            pass

    return durationList
