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


'''
ref: https://gist.github.com/bistaumanga/6023716
and http://tinyheero.github.io/2016/01/03/gmm-em.html
'''

# INIT_METHOD: KMEAN++, RANDOM, CENTER_LIST

class GMM_EM:
    """
    @var K_set: a list of K to try kmeans
    @var data: a set of data points from 2D synthetic Gausssian mixture 
               one data point is a tuple [label, x1,x2]
    @ var rep_count: given a K, the number of repetitive run of Kmeans
    """

    def __init__(self,X, k_list,data,K,epsilon,dim):
        self.k_list = k_list
        self.data = data
        
        self.N = len(data)  ## count of data points
        self.dim = dim  # dimension of data
        self.X = X
       
        self.K = K
        self.lamda = 0.2  # parameter for smoothing MLE

        self.epsilon  = epsilon  # threshold to stop
        self.max_iter = 5
        self.vars = namedtuple('vars',['mu','sigma','pi','log_likelihoods','member_mat'])



    '''
    in the maximizaiton process, we need to maximize:
    P(X| mu, sigma, pi) = sum over K (pi_k * N(X|mu_k,signma_k^2))
    then cost func: In (P(X| mu, sigma, pi))
    '''
    def ojective_func(self,center_dict,cluster_dict):
        J = 0.0
        for label,center in center_dict.iteritems():
            # distance within class
            point_set = cluster_dict[label]
            J += kmeans_utils.norm_to_center(point_set,center)
        return J


    '''
    given data points[N,K], return the probablilty of it belonging 
    to a normal distribution
    @return probability array[n]
    P(x | N(mu, sigma, pi)), pi is the weight for each Gaussian
    ref: http://stackoverflow.com/questions/11615664/multivariate-normal-density-in-python
    '''
    def get_prob(self,mu,sigma,pi):

        prob = np.empty((self.N))
        var = multivariate_normal(mu,sigma)
       
        for n in range(self.N):
            # for each of the data point, do...
            prob[n]= var.pdf(self.X[n,:])
           
            prob[n] = prob[n] * pi     
        return prob


    '''
    init K Gaussians using kmeans++
    return mu, sigma as matrix for K centers
    @ mu: array[K,dim]
    @ sigma: list[K] of array[dim*dim]
   
    '''
    def init_k_EM(self,K):
        points = self.X
        mu = list()
        
        # inti mu use Kmeans++ 
        center_dict = self.init_k_kmeans_plus(K)
        if self.dim==2:
            for label, center in center_dict.iteritems():
            # convert tuple to array
                mu.append(np.array((center[1],center[2])))
        else:
            for label, center in center_dict.iteritems():
            # convert tuple to array
                mu.append(np.array((center)))

        sigma = [np.identity(self.dim)]* K
        return mu,sigma


    '''
    run EM one time with a given K for Gaussian Mixture Models
    clusters are represented as a dict: {label: set of [x1,x2]}
    cluster centers are a dict of {label: center loc[x1,x2]}
    mean intialization uses Kmeans++ init
    covariance intialization uses identity matrix
    @ return J, and cluster_dict
    '''
    def EM_GMM(self,K,classes=None):
        data = self.data
        points = self.X
        N = self.N   # N data points
        dim = self.dim
        mu,sigma = self.init_k_EM(K)
        pi = [1.0/K] * K # initial priors
        # soft number of data points of each Gaussians, by summing up member matrix
        #  pi[k] = 1. / N * N_ks[k]
        N_k = [1.0/K] *K  
        # membership matrix: [N,k], the probability of each point 
        member_mat= np.zeros((self.N,K))
         ### log_likelihoods
        log_likelihoods = list()
        loss_list = list()
        iter_count = 0
        while len(log_likelihoods)< self.max_iter:
            print("------- iter_count----------",iter_count)      
            # E-step
            print("------- E step --------------")
            member_mat,log_likelihoods,N_k = self.Expectation(K,pi,mu,sigma,member_mat,log_likelihoods,N_k)
            print("------- M step --------------")
            # M-step
            pi,mu,sigma,log_likelihoods = self.Maximization(K,pi,mu,sigma,member_mat,log_likelihoods,N_k)
            iter_count = iter_count+1

            # get 0/1 loss

            # check convergence
            if len(log_likelihoods) < 2 : 
                continue
            if np.abs(log_likelihoods[-1] - log_likelihoods[-2]) < self.epsilon: 
                break
        
        print("final iter_count:",iter_count)

        self.vars.mu=mu
        self.vars.sigma = sigma
        self.vars.pi = pi
        self.vars.log_likelihoods = log_likelihoods
        self.vars.member_mat = member_mat  # responsibility matrix
        return self.vars

    '''
    given final results from GMM_EM, assign each point to its most likely cluster
    return: array[N] index represents data point idx, value represents cluster index
    '''
    def point_assinment(self,vars):
        mu = vars.mu
        member_mat = vars.member_mat
        assignment_mat = np.empty((self.N))
        for n in range(self.N):
            assignment_mat[n]= np.argmax(member_mat[n],axis = 0)
           
        return assignment_mat

    
    '''
    assign each point to existing Gaussians
    @return membership mat, equivalent to kmeans cluster hard 0/1 assignment
    '''
    def Expectation(self,K,pi,mu,sigma,member_mat,log_likelihoods,N_k):
        X = self.X
        for k in range(K):
            member_mat[:,k] = self.get_prob(mu[k],sigma[k],pi[k])
            # print("-----memeber_mat--------",member_mat[:,k])
        # compute likelihood
        log_likelihood = np.sum(np.log(np.sum(member_mat, axis = 1)))
        print("---likelihood in E step-----",log_likelihood)
        log_likelihoods.append(log_likelihood)
        # normalize membership matrix, and get the final membership(respobsibility) matrix
        member_mat = (member_mat.T / np.sum(member_mat, axis = 1)).T
        # soft number of data points of each Gaussians, by summing up member matrix
        # corresponds to number of points of each cluster in kmeans
        N_k = np.sum(member_mat, axis = 0)
        print("number of data points of each Gaussians:N_k:",N_k)
        return member_mat,log_likelihoods,N_k

    '''
    using responsibility matrix (member_mat), and the "soft" number of points 
    assigned to each class (N_k), estimate new center[mu, sigma] and prior pi
    based on MLE or MAP
    '''
    def Maximization(self,K,pi,mu,sigma,member_mat,log_likelihoods,N_k):
        X = self.X
        I = np.identity((self.dim))
        # sigma is a list of array[dim,dim]
        for k in range(K):
            mu[k] =np.sum(member_mat[:, k] * X.T, axis = 1).T / N_k[k]
            # when update sigma(array[dim,dim]), use MAP instead of MLE
            # sigma = MLE_sigma * (1-lamda) + lamda * I
            sigma_temp = np.zeros((self.dim,self.dim))
            
            for n in range(self.N):
                # this gives MLE estimates
                sigma_temp += member_mat[n,k] * np.outer(self.X[n,:]-mu[k],self.X[n,:]-mu[k])
            sigma_temp = (sigma_temp * (1-self.lamda) + self.lamda * I) /N_k[k]
            # normailize new prior
            pi[k] = 1. / self.N * N_k[k]
            # sigma.append(sigma_temp)
            sigma[k] = sigma_temp

        return pi,mu,sigma,log_likelihoods

    '''
    get classificaiton error (0/1) loss for GMM EM
    '''
    def get_EM_loss(self,classes,mu,membermat):
        loss = 0.0
        point_assinment(self,vars)
        for fakelabel,center in center_dict.iteritems():
            label_count = dict()
            # truelabel_center = self.get_label(classes,center)
            # check every member of this cluster, if its label coresponds to true label
            for point in cluster_dict[fakelabel]:
                truelabel_point = self.get_label(classes,point)
                label_count[truelabel_point]= label_count.get(truelabel_point,0)+1
            # get the max count of a certain label in the cluster
            max_count =label_count[max(label_count,key = label_count.get)]
            # print("max_coutn-----",max_count)
            # print("label_count--",label_count)
            # print("-----loss every iterator----",len(cluster_dict[fakelabel])-max_count)
            loss = loss +(len(cluster_dict[fakelabel])-max_count)
        print("----loss-------",loss)
                 
        return loss



    def plot_EM_llh(self,likelihoods,filename):
        fig = plt.figure(figsize=(12,8))
        plt.plot(likelihoods)
        plt.savefig(filename)
        

    '''
    for plotting EM
    ref: http://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib
    '''
    def eigsorted(self,cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    '''
    plot K clusters using multiple colors
    ref: http://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html
    '''
    def plot_cluster_bycolor(self,k_list,vars):
        print("---begin Plotting---------------")
        
        fig = plt.figure(figsize=(8,6))
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
        cmap = plt.get_cmap('jet')
        nstd = 2
        # loop a list of different K
        for i in range(len(k_list)):
            ax = fig.add_subplot(1,1,i+1)
            k = k_list[i]
            # choose k colors from color map
            colors = cmap(np.linspace(0, 1, k))
            # run EM_GMM
            assignment_mat = self.point_assinment(vars)
           
            for k,col in zip(range(k),colors):
                # my_members = k_means_labels == k
                cluster_center = vars.mu[k]
                # make points represented in n * 2 array [x1,x2]
                # point_array =kmeans_utils.points_to_array(cluster_dict[k],2)
                inx = assignment_mat[:]==k
                # print("-------inx------------",inx)
                point_array = self.X[inx]
                # print("----pointarray---",point_array)
                cov = vars.sigma[k]
                vals, vecs = self.eigsorted(cov)
                theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                w, h = 2 * nstd * np.sqrt(vals)

                ax.plot(point_array[:,0],point_array[:,1],'w',
                markerfacecolor = col,marker='.')
                ax.plot(cluster_center[0],cluster_center[1],'o',
                markerfacecolor = col,markeredgecolor='k',markersize=6)
                ell = Ellipse(xy=(cluster_center[0],cluster_center[1]),width=w, height=h,
              angle=theta, color='black')
                ell.set_facecolor('none')
                ax.add_artist(ell)
            ax.set_title('k=%d'%(k+1))
            ax.set_xticks(())
            ax.set_yticks(())
            # plt.text(-3.5,1.8,'k=%d'%k)
        plt.savefig('GMM_EM.png')


    
  

    '''
    run Kmeans one time with a given K
    clusters are represented as a dict: {label: set of [x1,x2]}
    cluster centers are a dict of {label: center loc[x1,x2]}
    # INIT_METHOD: KMEANS_PLUS, RANDOM, CENTER_LIST
    @ return J, and cluster_dict
    '''
    def kmeans(self,K,INIT_METHOD,max_iter = None,centers =None, classes=None):
        data = self.data
        J = float('inf')
        diff = float('inf')
        cluster_dict = dict()
        center_dict = dict()
        # class label : 0,1,2,....K
        labels=list(range(K))
        dim = self.dim
        loss_list = list()
        # Init: randomly pick K points without replacement as centers
        if INIT_METHOD == 'RANDOM':
            centers_list = random.sample(data,K)
            # print("random centers:",centers_list)
            center_dict = dict(zip(labels,centers_list)) 
            
        elif INIT_METHOD == 'KMEANS_PLUS':
            center_dict =  self.init_k_kmeans_plus(K)
        elif INIT_METHOD == 'CENTER_LIST':
            # read centers from BBC.centers
            centers_list = centers
            center_dict = dict(zip(labels,centers_list)) 
        else:
            raise Exception("choose a proper k initialization method")


        # print("random centers:",centers_list)
        # center_dict = dict(zip(labels,centers_list))    
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
                cluster_center = kmeans_utils.cluster_center(point_set,dim)
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
            # calculate 0/1 loss
            if classes!=None:
                loss_list.append(self.get_loss(classes,center_dict,cluster_dict))

            if diff>0.0:
            # clear cluster_dict
                for label,val in cluster_dict.iteritems():
                    print("end of iter label",label,"val",len(val))
                    cluster_dict[label].clear() 
            if iter_count==max_iter: break          

        print("final iter_count:",iter_count)
        if classes!=None:

            self.plot_loss(loss_list)
        return J,center_dict,cluster_dict


    '''
    plot classificaiton error (0/1) loss
    '''
    def plot_loss(self,loss_list):
        fig = plt.figure(figsize=(8,6))
        plt.plot(loss_list)
        plt.savefig('kmean_loss.png')
       

    '''
    get classificaiton error (0/1) loss
    '''
    def get_loss(self,classes,center_dict,cluster_dict):
        loss = 0.0

        for fakelabel,center in center_dict.iteritems():
            label_count = dict()
            # truelabel_center = self.get_label(classes,center)
            # check every member of this cluster, if its label coresponds to true label
            for point in cluster_dict[fakelabel]:
                truelabel_point = self.get_label(classes,point)
                label_count[truelabel_point]= label_count.get(truelabel_point,0)+1
            # get the max count of a certain label in the cluster
            max_count =label_count[max(label_count,key = label_count.get)]
            # print("max_coutn-----",max_count)
            # print("label_count--",label_count)
            # print("-----loss every iterator----",len(cluster_dict[fakelabel])-max_count)
            loss = loss +(len(cluster_dict[fakelabel])-max_count)
        print("----loss-------",loss)
                 
        return loss

    '''
    given a point (tuple), return its true label
    classes: dict: doc id -> class label
    '''
    def get_label(self,classes,point):
        data = self.data   # list of tuples: X[docid-1] = [....dim]
        docid = data.index(point)+1
        return classes[docid]


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





def main():
    filename = "2DGaussianMixture.csv"
    GM_set = processDoc.read_csv(filename)
    # for item in GM_set:
    #     print item
    print(len(GM_set))
    epsilon = 0.001
    k_set = set([2,3,4,10,15,20])
    k_list = [2,3,4,10,15,20]
    K = 3
    rep_count = 100
    dim = 2
    # convert data to array representation
    X = kmeans_utils.points_to_array(GM_set,dim)
    GE = GMM_EM(X,k_list,GM_set,K,epsilon,dim)
    GMM_vars = GE.EM_GMM(K)

    # print("----likelihood-----",GMM_vars.log_likelihoods)
    # print("------final mu-------",GMM_vars.mu)

    # GE.plot_EM_llh(GMM_vars.log_likelihoods)
    # GE.plot_cluster_bycolor([K],GMM_vars)


    # print("begin K-means")
    # J,center_dict,cluster_dict = lly.kmeans(K)
    # print("object func",J)
    # for label,center in center_dict.iteritems():
    #     print(label,center)

    print("begin K-means ++")
    J,center_dict,cluster_dict = GE.kmeans(K,'KMEANS_PLUS')
    print("object func",J)
    for label,center in center_dict.iteritems():
        print(label,center)

   
    # lly.plot_cluster_bycolor(k_list)

    # ---------plot Kmeans with different K---------
    # run Lylod for 20 times with K = 3

    # lly.rep_kmeans(K,rep_count)

    # lly.plot_repetitive_kmeans()
    # lly.plot_repetitive_kmeans_plus()



if __name__ == "__main__":
    main()
