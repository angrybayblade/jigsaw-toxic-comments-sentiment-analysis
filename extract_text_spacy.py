import pandas as pd
import numpy as np


from spacy.lang.en import English
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
import spacy

from notifyme import notify
from tqdm import tqdm

import sys

NLP = spacy.load('en_core_web_sm')

LEMMATIZE = lambda x:" ".join([i.lemma_ for i in NLP(x) if not i.is_stop])
df = pd.read_csv(f"./{sys.argv[1]}")


lemma = []
count = 0

print (df.comment_text.isna().sum())

for comment in tqdm(df.comment_text.values):
    if type(comment) == float:
        lemma.append("spam")
    else:
        comment = LEMMATIZE(comment)
        lemma.append(comment)

print (count)

df['comment_text'] = lemma
df.to_csv(f"./{sys.argv[2]}",index=False)

notify.success()