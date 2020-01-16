import pandas as pd
import numpy as np
import regex as re


from spacy.lang.en import English
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
import spacy

from notifyme import notify
from tqdm import tqdm

import sys

NLP = spacy.load('en_core_web_sm')

SYMBOL_FILTER = re.compile("[!@#%:;,\.?\'\"]")
CLEAN_UP = re.compile("[^a-z]")
WHITESPCAE_FILTER = re.compile("  *")
LEMMATIZE = lambda x:" ".join([i.lemma_ for i in NLP(x) if not i.is_stop])

def clean_sentence(x):
    x = " ".join(WHITESPCAE_FILTER.sub(" "," ".join(CLEAN_UP.sub(" "," ".join(SYMBOL_FILTER.sub("",x.lower()).split())).split())).split())
    return x



df = pd.read_csv(f"./{sys.argv[1]}")
df['comment_text'] = df.comment_text.apply(clean_sentence)

lemma = []

for comment in tqdm(df.comment_text.values):
    lemma.append(LEMMATIZE(comment))


df['comment_text'] = lemma
df.to_csv(f"./{sys.argv[2]}",index=False)

notify.success()