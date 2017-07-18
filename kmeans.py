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
import random

import processDoc
import kmeans_utils
# from processDoc import *


class Lloyd:
    """
    @var K_set: a list of K to try kmeans
    @var data: a set of data points from 2D synthetic Gausssian mixture 
               one data point is a tuple [label, x1,x2]
    @ var rep_count: given a K, the number of repetitive run of Kmeans
    """

    def __init__(self,k_list,data,rep_count,K):
        self.k_list = k_list
        self.data = data
        self.N = len(data)  ## count of data points
        self.rep_count = rep_count
        self.K = K

    '''
    @ center_dict : label: center[x1,x2]
    @ cluster_dict: label: set of points[label, x1, x2]
    '''
    def ojective_func(self,center_dict,cluster_dict):
        J = 0.0
        for label,center in center_dict.iteritems():
            # distance within class
            point_set = cluster_dict[label]
            J += kmeans_utils.norm_to_center(point_set,center)
        return J
    


    '''
    run Kmeans one time with a given K
    clusters are represented as a dict: {label: set of [x1,x2]}
    cluster centers are a dict of {label: center loc[x1,x2]}
    @ return J, and cluster_dict
    '''
    def kmeans(self,K):
        data = self.data
        J = float('inf')
        diff = float('inf')
        cluster_dict = dict()
        center_dict = dict()
        # class label : 0,1,2,....K
        labels=list(range(K))
        # Init: randomly pick K points without replacement as centers
        centers_list = random.sample(data,K)
        print("random centers:",centers_list)
        center_dict = dict(zip(labels,centers_list))    
        iter_count = 0

        while diff>0:
            print("------- iter_count----------",iter_count)         
            # E- step , assigning points to K centers
            for point in data:
                shortest = float('inf')        
                # current cluster lable of this data point
                cur_center = 999
                for label,center in center_dict.iteritems():
                    dist_temp=kmeans_utils.l2_norm(point, center)
                 #   print("dist_temp",dist_temp)
                    # dist_temp_list.append(utils.l2_norm(point, center))
                # shortest = min(dist_temp_list)
                #     # argmin
                # index = dist_temp_list.index(shortest)
                    if dist_temp<shortest:
                        shortest = dist_temp
                        cur_center = label
                # print("cur_center",cur_center,"of point:",point)
                # assign current point to a cluster
                if cluster_dict.has_key(cur_center):
                    cluster_dict[cur_center].add(point)
                else:
                    temp_set = set()
                    temp_set.add(point)
                    cluster_dict[cur_center] = temp_set
                           
            #  M- step, reassign new centers
            for label,point_set in cluster_dict.iteritems():
                cluster_center = kmeans_utils.cluster_center(point_set)
                center_dict[label] = cluster_center
                # print("label in cluster_dict, and its center",label,cluster_center )
                # print("label in cluster_dict, and len of pointset",label,len(point_set) )

            # get objective value j
            J_new = self.ojective_func(center_dict,cluster_dict)
            print("J_new:",J_new,"J",J,"iter_count",iter_count)
            diff = J-J_new
            print("----diff----",diff)
            J = J_new
            iter_count = iter_count+1
            if diff>0.0:
            # clear cluster_dict
                for label,val in cluster_dict.iteritems():
                    print("end of iter label",label,"val",len(val))
                    cluster_dict[label].clear()           

        print("final iter_count:",iter_count)
        return J,center_dict,cluster_dict


    '''
    run Kmeans++ one time with a given K
    clusters are represented as a dict: {label: set of [x1,x2]}
    cluster centers are a dict of {label: center loc[x1,x2]}
    @ return J, and cluster_dict
    '''
    def kmeans_plus(self,K):
        data = self.data
        J = float('inf')
        diff = float('inf')
        cluster_dict = dict()
        center_dict =  self.init_k_kmeans_plus(K)
        # class label : 0,1,2,....K
        labels=list(range(K))
        # Init: Pick the first cluster center u1 uniformly at random from the data 
        iter_count = 0
        # initial assignment of cluster
        while diff>0:
            print("------- iter_count----------",iter_count)         
            # E- step , assigning points to K centers
            for point in data:
                shortest = float('inf')
                dist_temp_list = list()          
                # current cluster lable of this data point
                cur_center = 999
                for label,center in center_dict.iteritems():
                    dist_temp=kmeans_utils.l2_norm(point, center)
 
                # index = dist_temp_list.index(shortest)
                    if dist_temp<shortest:
                        shortest = dist_temp
                        cur_center = label
                # print("cur_center",cur_center,"of point:",point)
                # assign current point to a cluster
                if cluster_dict.has_key(cur_center):
                    cluster_dict[cur_center].add(point)
                else:
                    temp_set = set()
                    temp_set.add(point)
                    cluster_dict[cur_center] = temp_set
                           
            #  M- step, reassign new centers
            for label,point_set in cluster_dict.iteritems():
                cluster_center = kmeans_utils.cluster_center(point_set)
                center_dict[label] = cluster_center
                # print("label in cluster_dict, and its center",label,cluster_center )
                # print("label in cluster_dict, and len of pointset",label,len(point_set) )

            # get objective value j
            J_new = self.ojective_func(center_dict,cluster_dict)
            print("J_new:",J_new,"J",J,"iter_count",iter_count)
            diff = J-J_new
            print("----diff----",diff)
            J = J_new
            iter_count = iter_count+1
            # clear cluster_dict
            if diff>0.0:
                for label,val in cluster_dict.iteritems():
                    print("end of iter label",label,"val",len(val))
                    cluster_dict[label].clear()           

        print("final iter_count:",iter_count)
        return J,center_dict,cluster_dict



    '''
    initialize cluster assignment for Kmeans ++
    '''
    def init_k_kmeans_plus(self,K):
        data = self.data
        center_dict = dict()
        # class label : 0,1,2,....K
        labels=list(range(K))
        # Init: Pick the first cluster center u1 uniformly at random from the data 
        # centers_list = random.sample(data,K)
        centers_list = list()
        first_center = random.sample(data,1)
        centers_list.append(first_center)
        print("first center:",first_center)  
        center_dict[labels[0]] = first_center[0]

        # initial assignment of cluster
        for j in range(1,K):
            print("--------- j-------------",j)   
            dist_temp_dict = dict()
           # list of D^2 for each data point
            for point in data:
                
                
                shortest = float('inf')         
                # current cluster lable of this data point
                cur_center = 999
                # loop through previous existing j+1 cluster
                for label,center in center_dict.iteritems():
                    dist = float("inf")
                    # print(label,center)
                    # print("data point",point)
                    dist=kmeans_utils.l2_norm(point, center)
                    # print("dist_temp",dist)
                    if dist<shortest:
                        shortest = dist
                        cur_center = label
                # print("shortest dist",shortest)
                dist_temp_dict[point] = shortest
                
            new_center = kmeans_utils.random_pick_prob(dist_temp_dict)
            # add new center to center_dict
            center_dict[labels[j]]= new_center
            
            print("---- intial K centers-------")
            for label,point in center_dict.iteritems():
                print(label,point)

        return center_dict


    

    '''
    run Kmeans of a given K for n times and choose the best resulting 
    clustering given the objective J
    return the best estimation
    '''
    def best_rep_kmeans(self,K,n):
        J = float("inf")      
        for i in range(n):
            J_temp,center_dict,cluster_dict = self.kmeans(K)
            if J_temp<J:
                J = J_temp
                best_center_dict,best_cluster_dict =center_dict,cluster_dict
        return J,best_center_dict,best_cluster_dict


    '''
    run Kmeans of a given K for n times and choose the best resulting 
    clustering given the objective J
    return summary stat of all estimations
    '''
    def rep_kmeans(self,K,n):
       
        J_set = set() 
        #  a set of cluster centers
        center_list= list()
        for i in range(n):
            J_temp,center_dict,cluster_dict = self.kmeans(K)
            J_set.add(J_temp)
            center_list.extend(center_dict.values())
               
        min_J, max_J, mean_J, sd_J = kmeans_utils.sum_stat(J_set)   
        # print("center_list",center_list)
        print("min_J:",min_J,"max_J",max_J,"mean_J",mean_J,"sd_J",sd_J)      
        return center_list


    '''
    run Kmeans++ of a given K for n times and choose the best resulting 
    clustering given the objective J
    return summary stat of all estimations
    '''
    def rep_kmeans_plus(self,K,n):
        
        J_set = set() 
        #  a set of cluster centers
        center_list= list()
        for i in range(n):
            J_temp,center_dict,cluster_dict = self.kmeans_plus(K)
            J_set.add(J_temp)
            center_list.extend(center_dict.values())
               
        min_J, max_J, mean_J, sd_J = kmeans_utils.sum_stat(J_set)   
        # print("center_list",center_list)
        print("min_J:",min_J,"max_J",max_J,"mean_J",mean_J,"sd_J",sd_J)      
        return center_list

    
    '''
    plot K clusters using multiple colors
    ref: http://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html
    '''
    def plot_cluster_bycolor(self,k_list):
        print("---begin Plotting---------------")
        
        fig = plt.figure(figsize=(12,6))
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
        cmap = plt.get_cmap('jet')
        # loop a list of different K
        for i in range(len(k_list)):
            ax = fig.add_subplot(2,3,i+1)
            k = k_list[i]
            # choose k colors from color map
            colors = cmap(np.linspace(0, 1, k))
            # run Kmeans
            J,center_dict,cluster_dict = self.kmeans(k)
           
            for k,col in zip(range(k),colors):
                # my_members = k_means_labels == k
                cluster_center = center_dict[k]
                # make points represented in n * 2 array [x1,x2]
                point_array =kmeans_utils.points_to_array(cluster_dict[k],2)

                ax.plot(point_array[:,0],point_array[:,1],'w',
                markerfacecolor = col,marker='.')
                ax.plot(cluster_center[0],cluster_center[1],'o',
                markerfacecolor = col,markeredgecolor='k',markersize=6)
            ax.set_title('k=%d'%(k+1))
            ax.set_xticks(())
            ax.set_yticks(())
            # plt.text(-3.5,1.8,'k=%d'%k)
        plt.savefig('kmeans.png')


    '''
    plot all centers with K = 3, repeated Kmeans for 20 times
    ref: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html 
   
    '''
    def plot_repetitive_kmeans(self):
        print("---begin Plotting---------------")   
        K = self.K
        data = self.data
        rep_count = self.rep_count
        
        fig = plt.figure(figsize=(12,8))
        center_list = self.rep_kmeans(K,rep_count)
        # convert center_list to array
        center_array = np.asarray(center_list)
        
        point_array = kmeans_utils.points_to_array(data,2)
        plt.plot(point_array[:,0],point_array[:,1],'w',
        marker ='.',markerfacecolor = 'grey')
        plt.plot(center_array[:,0],center_array[:,1],'o',
        marker ='.',markerfacecolor = 'black',markersize=12)
           
            # plt.text(-3.5,1.8,'k=%d'%k)
        plt.savefig('rep_kmeans.png')


    '''
    plot all centers with K = 3, repeated Kmeans for 20 times
    ref: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html 
   
    '''
    def plot_repetitive_kmeans_plus(self):
        print("---begin Plotting---------------")   
        K = self.K
        data = self.data
        rep_count = self.rep_count
        
        fig = plt.figure(figsize=(12,8))
        plt.clf()
        center_list = self.rep_kmeans_plus(K,rep_count)
        # convert center_list to array
        center_array = np.asarray(center_list)
        
        point_array = kmeans_utils.points_to_array(data,2)
        plt.plot(point_array[:,0],point_array[:,1],'w',
        marker ='.',markerfacecolor = 'grey')
        plt.plot(center_array[:,0],center_array[:,1],'o',
        marker ='.',markerfacecolor = 'black',markersize=12)
           
            # plt.text(-3.5,1.8,'k=%d'%k)
        plt.savefig('rep_kmeans_plus.png')







def main():
    filename = "2DGaussianMixture.csv"
    GM_set = processDoc.read_csv(filename)
    # for item in GM_set:
    #     print item
    print(len(GM_set))
    k_set = set([2,3,4,10,15,20])
    k_list = [2,3,4,10,15,20]
    K = 3
    rep_count = 20
    lly = Lloyd(k_list,GM_set,rep_count,K)


    # print("begin K-means")
    # J,center_dict,cluster_dict = lly.kmeans(K)
    # print("object func",J)
    # for label,center in center_dict.iteritems():
    #     print(label,center)

    # print("begin K-means ++")
    # J,center_dict,cluster_dict = lly.kmeans_plus(K)
    # print("object func",J)
    # for label,center in center_dict.iteritems():
    #     print(label,center)

   
    # lly.plot_cluster_bycolor(k_list)

    # ---------plot Kmeans with different K---------
    # run Lylod for 20 times with K = 3

    # lly.rep_kmeans(K,rep_count)

    lly.plot_repetitive_kmeans()
    # lly.plot_repetitive_kmeans_plus()

    




if __name__ == "__main__":
    main()
