import pandas as pd
import csv
import re
import nltk
import emoji
import numpy as np
import unicodedata
from contractions import contractions_dict
# from contractions import CONTRACTION_MAP
stopword_list = nltk.corpus.stopwords.words('english')
# just to keep negation if any in bi-grams
stopword_list.remove('no')
stopword_list.remove('not')


# def expand_contractions(text, contraction_mapping=contractions_dict):
#     contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
#                                       flags=re.IGNORECASE | re.DOTALL)
#
#     def expand_match(contraction):
#         match = contraction.group(0)
#         first_char = match[0]
#         expanded_contraction = contraction_mapping.get(match) \
#             if contraction_mapping.get(match) \
#             else contraction_mapping.get(match.lower())
#         expanded_contraction = first_char + expanded_contraction[1:]
#         return expanded_contraction
#
#     expanded_text = contractions_pattern.sub(expand_match, text)
#     expanded_text = re.sub("'", "", expanded_text)
#     return expanded_text

def expand_contractions(text, contraction_mapping=contractions_dict):
  contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                    flags=re.IGNORECASE | re.DOTALL)

  def expand_match(contraction):
      match = contraction.group(0)
      first_char = match[0]
      expanded_contraction = contraction_mapping.get(match) \
          if contraction_mapping.get(match) \
          else contraction_mapping.get(match.lower())
      expanded_contraction = first_char + expanded_contraction[1:]
      return expanded_contraction


  try:
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
  except:
    return text
  return expanded_text

def remove_mentioning(text):
    text = re.sub(r'(^)@\w+', r'\1', text)
    text = re.sub(r'(\s)@\w+', r'\1', text)
    return text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_url(text):
    text = re.sub(r"(http |http).*$", '', text)
    return text

import spacy
nlp = spacy.load('en_core_web_sm', parse=True, tag=True, entity=True)
def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


from wiki_ru_wordnet import WikiWordnet
wikiwordnet = WikiWordnet()

from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()



from spacy_stanza import StanzaLanguage
import stanza

stanza.download('en')
stanza_nlp = stanza.Pipeline('en')
snlp = stanza.Pipeline(lang="en")
nlp = StanzaLanguage(snlp)



def remove_url(text):
    text = re.sub(r"(http |http).*$", '', text)
    return text

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def find_emoji(text):
    emojis_in_comment = [char for char in text if char in emoji.UNICODE_EMOJI]
    return emojis_in_comment


def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'

    def replace(old_word):
        if wikiwordnet.get_synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        return replace(new_word) if new_word != old_word else new_word

    correct_tokens = [replace(word) for word in tokens]
    text = ' '.join(correct_tokens)
    return text




# def lemmatisaze_document(doc):
#
#     nlp = StanzaLanguage(snlp)
#     doc = nlp(doc)
#
#     filtered_tokens = [token.lemma_ for token in doc]
#     doc = ' '.join(filtered_tokens)
#     return doc


def normalize_corpus(corpus, contraction_expansion=True, accented_char_removal=True,
                     special_char_removal=True, remove_digits=True,
                     repeated_characters_remover=True, text_lower_case=True,
                     stop_words_remover = True, stopwords=stopword_list, text_lemmatization=True, mentioning_remover = True, stopword_removal =True):

    normalized_corpus = []
    list_of_emoji = []
    timecode_list = []
    timecodes = []
    emoji = []

    # normalize each document in the corpus
    for doc in corpus:

        # remove extra newlines
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))

        # # remove extra whitespace
        # doc = re.sub(' +', ' ', doc)
        # doc = doc.strip()

        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)

        doc = remove_url(doc)
        # expand contractions
        if contraction_expansion:
            doc = expand_contractions(doc)

        # lemmatize text
        if text_lemmatization:
             doc = lemmatize_text(doc)
            
        # lowercase the text
        if text_lower_case:
             doc = doc.lower()
                
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        
        # remove mation of airline
        if mentioning_remover:
            doc = remove_mentioning(doc)
            
        if special_char_removal:
            # insert spaces between special characters to isolate them
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)

                
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
            
        # remove stopwords
        if stopword_removal:
             doc = remove_stopwords(doc, is_lower_case=text_lower_case)

        normalized_corpus.append(doc)
    return normalized_corpus