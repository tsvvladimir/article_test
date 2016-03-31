from __future__ import division
from __future__ import unicode_literals
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from numpy.random import randint
import logging
import sys
import operator
import math
from sklearn.multiclass import OneVsRestClassifier
from collections import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import wordnet as wn
import re
from itertools import chain
from sklearn.utils import shuffle
import nltk
import string
import os
from nltk.stem.porter import PorterStemmer
from string import maketrans
from sklearn.cluster import KMeans
import inspect
from multiprocessing import Process
from nltk import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime
import pickle
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.calibration import calibration_curve
import itertools as it
from numpy import linalg as LA

def diploma_res_print(foldname, volume, score, par=None):
    #print '(', volume, '; ', score * 100, ')'
    #print 'vol.append(',volume,')'
    #print 'score.append(',score * 100,')'
    if par == None:
        print (inspect.stack()[1][3] + '_' + foldname), volume, '; ', score
    else:
        print (inspect.stack()[1][3] + '_' + foldname), volume, '; ', score, ';', par
    #f.write(str(inspect.stack()[1][3]) + '(' + str(volume) + '; ' + str(score * 100)+ ')')

def diploma_random_sampling(dst_data, dst_target, src_data, src_target, n):
    idxes = []
    for i in range(0, n):
        idx = np.random.randint(0, len(src_data))
        dst_data = np.append(dst_data, src_data[idx])
        dst_target = np.append(dst_target, src_target[idx])
        idxes.append(idx)
    src_data = np.delete(src_data, idxes)
    src_target = np.delete(src_target, idxes)
    return (dst_data, dst_target, src_data, src_target)

def diploma_range_sampling(dst_data, dst_target, src_data, src_target, n):
    for i in n:
        dst_data = np.append(dst_data, src_data[i])
        dst_target = np.append(dst_target, src_target[i])
    src_data = np.delete(src_data, n)
    src_target = np.delete(src_target, n)
    return (dst_data, dst_target, src_data, src_target)