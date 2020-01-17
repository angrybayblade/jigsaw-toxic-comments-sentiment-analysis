import pandas as pd
import numpy as np
import regex as re

from notifyme import notify
from tqdm import tqdm

import sys

SYMBOL_FILTER = re.compile("[!@#%:;,\.?\'\"]")
CLEAN_UP = re.compile("[^a-z]")
WHITESPCAE_FILTER = re.compile("  *")

def clean_sentence(x):
    x = " ".join(WHITESPCAE_FILTER.sub(" "," ".join(CLEAN_UP.sub(" "," ".join(SYMBOL_FILTER.sub("",x.lower()).split())).split())).split())
    return x


df = pd.read_csv(f"./{sys.argv[1]}")

comments_text = []

for comment in tqdm(df.comment_text.values):
    comments_text.append(clean_sentence(comment))

df['comment_text'] = comments_text

df.to_csv(f"./{sys.argv[2]}",index=False)
notify.success()