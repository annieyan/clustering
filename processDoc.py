import numpy as np
import time
import sys
import collections
import math
import operator
import unicodedata as ucd  # deal with Latin char such as \xc2\xa37m
'''
usage
strange_str ='\xc2\xa37m'
    strange_str = strange_str.decode('latin1')
    print(ucd.name(strange_str[0]))
    print(strange_str.encode('latin-1').decode('utf-8'))
'''

'''
read inn 2D synthetic CSV data
'''
def read_csv(filename):
    i = 0
    GM_set = set()
    with open(filename,'r') as file:      
        for i, line in enumerate(file):
            # skip header
            if i<2:
                continue
            # parse line 
            GM_tuple = tuple()
            #GM_data_line = file.readline()
           # print GM_data_line
            fields = line.split(",")
            if len(fields)!=3:
                raise Exception("raw data has incorrect number of fields",GM_data_line)
            class_label =int(fields[0])
            x1 =  float(fields[1])
            x2 =  float(fields[2].strip())
            GM_tuple = class_label,x1,x2
            GM_set.add(GM_tuple)
            
    return GM_set

##################  process BBC news data ################################

'''
parse BBC .mtx
"termid docid frequency"
'''
def read_in_mtx(filename, has_header):
    """
    Reads in a data set
    @param filename: string - name of file (with path) read
    @param has_header: boolean - whether the file has a header
    """
    documents = dict()
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            # print("line----",line)
            if i < 2 and has_header:
                continue
            parse_in_mtx(line, documents)
    return documents


def parse_in_mtx(line, documents):
    """
    Reads in line from .mtx file into documents hash.
    @param line: string - line to parse
    @param documents: dict - hash of documents (doc id -> term id -> count)
    """
    # print("line",line)
    fields = line.split(" ")
    if len(fields) != 3:
        raise Exception("Matrix line has incorrect number of fields: " + line)
    
    term_id = int(fields[0])
    doc_id = int(fields[1])
    count = float(fields[2].rstrip())
    # print("termid, doc_id,count",term_id,doc_id,count)
    if not documents.has_key(doc_id):
        documents[doc_id] = dict()
    documents[doc_id][term_id] = count

'''
vector representation for a document using tfidf
tfidf_dict: {termid, docid: tfidf value})
return tuple {vector[V_size]}
'''
def doc_to_vec(documents,tfidf_dict):
    dim= len(tfidf_dict)
    doc_vec = list()
    doc_count = len(documents)
    docid_list = documents.keys()
    for docid in range(doc_count):
        temp_list = tuple()
        for termid in range(dim):
            # termid starts from 1
            if tfidf_dict[termid+1].has_key(docid+1):
                temp_list = temp_list+(tfidf_dict[termid+1][docid+1],)
            else:
                temp_list = temp_list+(0.0,)
        doc_vec.append(temp_list)
    return doc_vec


def parse_in_classes(line, documents):
    """
    Reads in line from .mtx file into documents hash.
    @param line: string - line to parse
    @param documents: dict - hash of documents (doc id -> class label)
    """
    # print("line",line)
    fields = line.split(" ")
    if len(fields) != 2:
        raise Exception("Matrix line has incorrect number of fields: " + line)   
    doc_id = int(fields[0])
    label = int(fields[1].rstrip())
    # print("doc_id,label",doc_id,label)
    if not documents.has_key(doc_id):
        documents[doc_id] = dict()
    documents[doc_id]= label


def parse_in_center(line):
    """
    Reads in line from .mtx file into documents hash.
    @param line: string - line to parse
    @param centers: tuple (x1,x2,x3....xn)
    """
    # print("line",line)
    fields = line.split(" ")
    center = tuple()
    dim = len(fields)
    if dim != 99:
        raise Exception("Matrix line has incorrect number of fields: " + line)   
    for i in range(dim):
        center= center + (float(fields[i]),)
    return center




'''
read in .terms file in BBC data
@ return {termid: term} note that termid starts from 1-99
ref: https://docs.python.org/3/library/codecs.html#standard-encodings
'''
def read_in_terms(filename, has_header):
    documents = dict()  # termid: term
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            # strip '\n'
            line = line.rstrip()          
            # print("line----",line)
            # line = line.decode('latin-1')
            # # line = line.('latin_1')
            # line = line.encode('latin-1').decode('utf-8')
            if i < 2 and has_header:
                continue
            # parse_in_line(line, documents)
            documents[i+1] = line
    return documents



'''
read in .classes file in BBC data
'''
def read_in_classes(filename, has_header):
    documents = dict()
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            # strip '\n'
            line = line.rstrip()
            # print("line----",line)
            
     
            if i < 2 and has_header:
                continue
            parse_in_classes(line, documents)
    return documents


'''
read in .centers file in BBC data
return a set of cluster centers : a list of tuples
'''
def read_in_centers(filename, has_header):
    centers = list()
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            # strip '\n'
            line = line.rstrip()
            # print("line----",line)
            if i < 2 and has_header:
                continue
            centers.append(parse_in_center(line))
    return centers

'''
ref: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
tf(t,d) = f(t,d) / max{f(w,d): w in d}
@ documents: documents[doc_id][term_id] = count
'''
def tf(documents):
    # {termid, docid : tf}
    tf_dict = dict()
    for docid, val in documents.iteritems():
        # max freq term in one doc

        max_freq = float(max(val.values()))
        for termid, count in val.iteritems():
            if not tf_dict.has_key(termid):
                tf_dict[termid] = dict()
            tf_dict[termid][docid] = float(float(count) / float(max_freq))        

    return tf_dict


'''
idf(t,D) = log{ N / |{d in D: t in d}|}
@ N: total number of documents in the corpus
@ |{d in D: t in d}|: number of documents where the
term t appears.  If the term is not in the corpus, 
this will lead to a division-by-zero. 
It is therefore common to adjust the denominator to {1+|\{d\in D:t\in d\}|}
It is the logarithmically scaled inverse fraction of the documents
that contain the word, obtained by dividing the total number of 
documents by the number of documents containing the term,
and then taking the logarithm of that quotient.

@ documents: documents[doc_id][term_id] = count
@ return {termid: idf value}
'''
def idf(documents):
    N = len(documents)
    # {term: value}
    idf_dict = dict()
    # number of documents where the term t appears.  
    # {termid: count of docs}
    doc_count = dict()
    # search for documents that contain a particular term
    for docid, val in documents.iteritems():
        # # max freq term in one doc
        # max_freq = float(max(val.values()))
        for termid, count in val.iteritems():
            doc_count[termid]=doc_count.get(termid,0.0)+1.0
    
    for termid, count in doc_count.iteritems():
        if count!=0:
            idf_dict[termid] = math.log(float(float(N)/float(count)))
        else:
            print("term count is 0 in a doc, termid",termid)
            idf_dict[termid] = float("inf")
    return idf_dict

'''
Convert the term-doc-frequency matrix into a term-doc-tfidf matrix
return @ tfidf_dict: {termid, docid: tfidf value}
return @ avg_tfidf_dict: {termid, class_label :  avg tiidf value}
'''
def tfidf(documents,bbc_classes):
    idf_dict = idf(documents)
    tf_dict = tf(documents)
    tfidf_dict = dict()
    avg_tfidf_dict = dict()  # termid, class label:  tfidf
    class_list = bbc_classes.values()
    class_count_dict = dict()   # class_label: count
    # class_count = len(bbc_classes)
    # get term-doc-freq matrix
    for docid, val in documents.iteritems():
        class_label = bbc_classes[docid]
        # if not avg_tfidf_dict.has_key[class_label]:
        #     avg_tfidf_dict[]
        for termid, count in val.iteritems():
            if not tfidf_dict.has_key(termid):
                tfidf_dict[termid] = dict()
            tfidf_dict[termid][docid]=tf_dict[termid][docid] * idf_dict[termid]
            if not avg_tfidf_dict.has_key(termid):
                avg_tfidf_dict[termid] =  dict()
            avg_tfidf_dict[termid][class_label] = avg_tfidf_dict[termid].get(class_label,0)+1
            # print("avg_tfidf_dict[termid][class_label]",avg_tfidf_dict[termid][class_label] )
            # print("class_label",class_label)
        class_count_dict[class_label]=class_count_dict.get(class_label,0)+1
    # print("len of avg_tfidf_dict",len(avg_tfidf_dict))
    # print("all terms in avg_tfidf_dict",avg_tfidf_dict.keys())
    # print("all vals in avg_tfidf_dict",avg_tfidf_dict.values())

    for termid,val in avg_tfidf_dict.iteritems():
        for label, val in val.iteritems():
            # print("avg_tfidf_dict[termid][class_label]",avg_tfidf_dict[termid][label])
            # print("class_count_dict[class_label]",class_count_dict[label])
            avg_tfidf_dict[termid][label] = float(float(avg_tfidf_dict[termid][label]) / float(class_count_dict[label]))

    return tfidf_dict,avg_tfidf_dict


'''
Convert the term-doc-frequency matrix into a avg term-doc-tfidf matrix
@ var : documents, idf_dict,tf_dict
@ var: bbc_classes :  (docid: label)
return @ tfidf_dict: {termid, docid: tfidf value}
'''
# def avg_tfidf(tfidf_dict,bbc_classes):
#     tfidf_dict = dict()
#     # get term-doc-freq matrix
#     for docid, val in documents.iteritems():
#         for termid, count in val.itertimes():
#             tfidf_dict[termid][docid]=tf_dict[termid][docid] * idf_dict[termid]
#     return tfidf_dict

'''
For each class Ci,
return @ sorted avg tfidf {label: (termid, avg tfidf)}
'''
def sort_avg_tfidt(term_freq_mtx,bbc_classes):
    idf_dict = idf(term_freq_mtx)
    tf_dict = tf(term_freq_mtx)
    tfidf_dict,avg_tfidf_dict = tfidf(term_freq_mtx,bbc_classes)
    #avg_tfidf_dict: {termid, class_label :  avg tfidf value}
    # reverse_avgtfidf = list()  # tuple(label,termid, avg tfidf)}
    reverse_avgtfidf_dict = dict()  # label: term, avg tfidf
    for termid, val in avg_tfidf_dict.iteritems():
        for label,avg_tfidf in val.iteritems():
            # reverse_avgtfidf.append((label,termid, avg_tfidf))
            if not reverse_avgtfidf_dict.has_key(label):
                reverse_avgtfidf_dict[label] = list()
            reverse_avgtfidf_dict[label].append((termid, avg_tfidf))
 
    sorted_tfidf = dict()    # sorted avg tfidf {label: (termid, avg tfidf)}
    for label, vals in reverse_avgtfidf_dict.iteritems():
        if not sorted_tfidf.has_key(label):
            sorted_tfidf[label] = list()
        # sort by the 2nd term: avg tfidf
        sorted_tfidf[label] = sorted(vals, key=operator.itemgetter(1),reverse=True)
    # for label, val in sorted_tfidf.iteritems():
    #     print("label",label)
    #     print("termid, avg tfidf",val)
    return sorted_tfidf

'''
given a termid, return term
'''
def get_term(termid,bbc_terms):
    return bbc_terms[termid]

'''
get class content given a label
business, entertainment, politics, sport, tech
'''
def get_class_by_label(label):
    
    class_dict = {0:"business",1:"entertainment",2:"politics",
    3:"sport",4:"tech"}
    class_name = class_dict[label]
    return class_name

##########################  main ###################################
def main():
    # filename = "2DGaussianMixture.csv"
    # GM_set = read_csv(filename)
    # # for item in GM_set:
    # #     print item
    # print(len(GM_set))
    mtx_file = "bbc.mtx"
    term_freq_mtx = read_in_mtx(mtx_file,True)
    # print("termid  values",term_freq_mtx.values())

    termfile = "bbc.terms"
    bbc_terms = read_in_terms(termfile,False)
    # print("term list",bbc_terms)

    class_file = "bbc.classes"
    bbc_classes = read_in_classes(class_file,False)
    print("bbc_classes",bbc_classes)

    center_file = "bbc.centers"
    centers = read_in_centers(center_file,False)
    # print("centers",centers)

    # For each class Ci,report the 5 terms with the highest AvgTfidf for the class 
    # sorted_tfidf:   {label:(termid,avg_tfidf)}
    sorted_tfidf = sort_avg_tfidt(term_freq_mtx,bbc_classes)
    for label, val in sorted_tfidf.iteritems():
        print("label:",get_class_by_label(label))
        i = 0
        while i<5:
            # label_term = 
            # print("val[0]",val[i])
            term = bbc_terms[val[i][0]]
            print("term, avg tfidf",term,val[i][1])
            i = i+1

   






if __name__=="__main__":
    main()