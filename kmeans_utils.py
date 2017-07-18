from __future__ import division
import numpy as np
import os
from collections import Counter
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
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
from numpy import random


'''
@ vec1 : point:[label, x1, x2]
@ vec2: center: [x1,x2] or [label, x1, x2]
'''
def l2_norm(vec1,vec2):
    l2_norm = 0.0
    dim = len(vec2)
    # assert(dim==len(vec2))
    if dim == 3:
    # skip the the index 0, which is class label
        for i in range(1,dim):
            l2_norm += math.pow(vec1[i]-vec2[i],2)
    elif dim ==2:
        for i in range(0,dim):
            l2_norm += math.pow(vec1[i+1]-vec2[i],2)
    # deal with BBC news data
    elif dim==99:
        for i in range(0,dim):
            l2_norm += math.pow(vec1[i]-vec2[i],2)
    else:
        for i in range(1,dim):
            l2_norm += math.pow(vec1[i]-vec2[i],2)    
    return l2_norm

'''
given a cluster of points, return the center point
works for 2d Gaussian data only
'''
def cluster_center(point_set,dim):
    if dim == 2:
        count = float(len(point_set))
        dim1 = 0.0
        dim2 = 0.0
        center = tuple()
        for point in point_set:
            dim1 += point[1]
            dim2 += point[2]
        dim1 = float(dim1/count)
        dim2 = float(dim2/count)
        return dim1,dim2
    else:
        
    # center = tuple()
        count = len(point_set)
    # dim1 = 0.0
    # dim2 = 0.0
    # center = tuple()
        point_array = np.empty((count,dim))
        i = 0
        for point in point_set:
            point_array[i,:] = np.array(point)
            i = i+1
        mean_array = np.empty((dim))
        # N * dim
        # point_array = np.asarray(point_set)
        # print("-----point_array------",point_array)
        mean_array = np.mean(point_array,axis = 0)
    # print("-----mean_array-----",mean_array.shape)
    # convert to tuple

    # for point in point_set:
    #     dim1 += point[1]
    #     dim2 += point[2]
    # dim1 = float(dim1/count)
    # dim2 = float(dim2/count)
        return tuple(mean_array)

'''
the sum distance of every point to its cluster center
'''
def norm_to_center(point_set, center):
    dist = 0.0
    for point in point_set:
        # skip label in point
        # point_temp = point[1:]
        dist+= l2_norm(point,center)

    return dist

'''
given a set of J, get summary stat
'''
def sum_stat(result_set):
    result_list = list(result_set)
    min_J = min(result_list)
    max_J = max(result_list)
    mean_J = np.mean(result_list)
    sd_J = np.std(result_list)
    return min_J, max_J,mean_J, sd_J


'''
given a dict of points with its shortest distance
used in Kmeans++
input : dict {point[x1,x2], l2_norm}
'''
def random_pick_prob(point_dict):
    point_list = list()
    dist_list = list()
    index_list = range(0,len(point_dict))
    for point,dist in point_dict.iteritems():
        point_list.append(point)
        dist_list.append(dist)
    # get probabilities
    norm = [float(i)/float(sum(dist_list)) for i in dist_list]
    # print("----- point_list---",point_list)
    chosen_index = random.choice(index_list, 1, replace=False, p=norm)
    print("chosen index",chosen_index[0])
    chosen_center = point_list[chosen_index[0]]
    print("pick next point:",chosen_center)
    return chosen_center


'''
given a set of points : set of tuples or list of tuples, 
and a dimension, 
return a n*dim array of points 
'''
def points_to_array(point_set,dim):
    n = len(point_set)
    point_array = np.empty([n,dim])
    i = 0
    if dim ==2:
        for point in point_set:
            # print("point",point)
            point_array[i,0] =  point[1]
            point_array[i,1] =  point[2]
            i = i+1
    else:
        for point in point_set:
            # print("point",point)
            for d in range(dim):
                point_array[i,d]=point[d]
            i = i+1

    # for pretty print np array    
    # np.set_printoptions(precision=3)
    # print("point array",point_array)
    return point_array