import re
import numpy as np
import scipy
import math
import random
from gensim import models
from nltk import sent_tokenize
from gensim.models.doc2vec import TaggedDocument

# Build the input for one text
def text2data(text,title,tag_id):
    text = text + title
    txt = sent_tokenize(text)
    words = []
    tags = []
    for sent in txt:
        words.append(re.findall(r"[\w*|\d*]\S*[\w|\d]|[,!?;()\d*\w]", sent))
        #tags.append("DOC_" + str(tag_id))
        tags.append(tag_id)
    return words,tags

def Tag_texts(texts,titles,tags):
    docs = []
    for i in range(len(texts)):
        wds , tagged = text2data(texts[i],titles[i],tags[i])
        for st in range(len(wds)):
            docs.append(TaggedDocument(words=wds[st], tags=[tagged[st]]))
    return docs 

def construct_matrix(model,idx):
    mat = []
    for id in idx:
        mat.append(model.docvecs[id].reshape(1, -1))
    return np.asarray(mat)