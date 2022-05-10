# coding=utf-8

import unicodedata
import numpy


def _replace_diacritics(s):
    # lambda x : x.replace(u'ø', u'o').replace(u'æ', u'ae'),
    # what about ü,ö,ä,...
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("utf-8")


def _replace_multi_spaces(s):
    while "  " in s:
        s = s.replace("  ", " ")
    return s


def _replace_double_quotation_marks(s):
    return s.replace('"', " ")


def _replace_trailing_non_alphanumeric_chars(s):
    while s and not s[-1].isalnum():
        s = s[0:-1]
    return s


def _correct_commas_and_periods(s):
    s = s.replace(" ,", ",")
    s = s.replace(" .", ".")
    s = s.replace(", ", ",")
    # s = s.replace(u'. ', u'.')
    return s


# def _replace_spelling_variations(s):
#    return s

purge_rules = [
    lambda x: x.replace(" l'", " l "),  # french example: 'L'Habitat'
    lambda x: x.replace("-", " "),
    lambda x: x.replace("(", " "),
    lambda x: x.replace(")", " "),
    # lambda x : x[4:] if x.startswith(u'the ') else x,
    lambda x: x.replace(" & ", " and "),
    lambda x: x.replace(" + ", " and "),
    lambda x: x.replace(";", " "),
    lambda x: x.replace("/", " "),
    lambda x: x.replace(",", " "),
]


def purge(s):
    for rule in purge_rules:
        s = rule(s)
    return s


steps = [
    lambda x: x.lower(),
    _replace_diacritics,
    _replace_multi_spaces,
    _replace_double_quotation_marks,
    _replace_trailing_non_alphanumeric_chars,
    _correct_commas_and_periods,
    purge,
    _replace_multi_spaces,
]


def harmonize(s):
    """
    Harmonization for company name matching inspired by
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.95.6455&rep=rep1&type=pdf
    """
    for step in steps:
        try:
            s = step(s)
        except Exception as e:
            raise Exception("Step failed: %s - %s" % (str(step), s), e)
    return s


_synonyms = {
    "&": "and",
    "ag": "aktiengesellschaft",
    "co": "company",
    "co.": "company",
    "corp": "corporation",
    "corp.": "corporation",
    "inc": "incorporated",
    "inc.": "incorporated",
    # u'inco': u'incorporated',
    "int": "international",
    "int.": "internatinal",
    "intl.": "international",
    "ltd": "limited",
    "ltd.": "limited",
    "pvt": "private",
}


def tokenize(s):

    # "Adding special case tokenization rules"
    # from https://spacy.io/usage/linguistic-features
    s = harmonize(s).strip()
    tokens = s.split(" ")

    # Spacy: use lemmatizer to put legal form surface forms into
    # lemmas (what I call synonyms here)
    # I imagine something like this: "Ltd." will be lemmatized as "limited".
    # https://spacy.io/usage/linguistic-features
    return numpy.unique([_synonyms.get(token, token) for token in tokens])
