import re
import pickle
import unicodedata
from tqdm import tqdm
from datetime import datetime, timezone
import pytz
import numpy as np
import pandas as pd
import simplemma
from nltk import ngrams
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from model import Review_Model

tmz=pytz.timezone('Europe/Berlin')

tqdm.pandas(desc="Example Desc")



data =pd.read_csv("../src/review.csv")


obj = Review_Model(records =data)
df = obj.main()

print(df.shape)
print(df.head(2))

df[df["tags"].str.contains("anti")][["text","tags","class"]]