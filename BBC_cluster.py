from __future__ import division
import numpy as np
import os
from collections import Counter
import nltk
import string
from collections import deque
from itertools import islice
import collections
import math
import argparse
import time
import json
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import re
import matplotlib.pyplot as plt
import itertools
import sys
import random

import processDoc
import kmeans_utils
from collections import namedtuple
import scipy
from scipy.stats import norm  # for Gaussian 
from scipy.stats import multivariate_normal
from matplotlib.patches import Ellipse
import GMM_EM
from GMM_EM import GMM_EM
# from processDoc import *


 

'''
use kmeans and EM to cluster BBC news data
'''
def main():
    #------------read files ------------------#
    mtx_file = "bbc.mtx"
    term_freq_mtx = processDoc.read_in_mtx(mtx_file,True)
    # print("termid  values",term_freq_mtx.values())

    termfile = "bbc.terms"
    bbc_terms = processDoc.read_in_terms(termfile,False)
    # print("term list",bbc_terms)

    class_file = "bbc.classes"
    bbc_classes = processDoc.read_in_classes(class_file,False)
    # print("bbc_classes",bbc_classes)

    # a set of tuples 
    center_file = "bbc.centers"
    centers = processDoc.read_in_centers(center_file,False)

    
    epsilon = 0.001
    dim = 99
    max_iter = 5
    
    k_list = [2,3,4,10,15,20]
    K = 5
    rep_count = 100
    figurename = 'BBC_EM_likelihood.png'

    tfidf_dict,_= processDoc.tfidf(term_freq_mtx,bbc_classes)

    data = processDoc.doc_to_vec(term_freq_mtx,tfidf_dict)
    X =kmeans_utils.points_to_array(data,dim)

    GE = GMM_EM(X,k_list,data,K,epsilon,dim)
    # kmeans
    print("--------kmeans---------------")
    # J,center_dict,cluster_dict = GE.kmeans(K,"CENTER_LIST",max_iter,centers,bbc_classes)

    # print("----likelihood-----",BBC_vars.log_likelihoods)
    # print("------final mu-------",BBC_vars.mu)
    # EM
    print("----------EM---------------")
    GMMvars = GE.EM_GMM(K)
    GE.plot_EM_llh(GMMvars.log_likelihoods,figurename)

    

    # print("begin K-means")
    # J,center_dict,cluster_dict = lly.kmeans(K)
    # print("object func",J)
    # for label,center in center_dict.iteritems():
    #     print(label,center)

    # print("begin K-means ++")
    # J,center_dict,cluster_dict = GE.kmeans_plus(K)
    # print("object func",J)
    # for label,center in center_dict.iteritems():
    #     print(label,center)

   
    # lly.plot_cluster_bycolor(k_list)

    # ---------plot Kmeans with different K---------
    # run Lylod for 20 times with K = 3

    # lly.rep_kmeans(K,rep_count)

    # lly.plot_repetitive_kmeans()
    # lly.plot_repetitive_kmeans_plus()

    

if __name__ == "__main__":
    main()
