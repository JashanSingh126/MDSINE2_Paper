from scipy.spatial import distance
import numpy as np


def braycurtis(u,v):
    return distance.braycurtis(u,v)

def jaccard(u,v):
    return distance.jaccard(u,v)

def euclidean(u,v):
    return distance.euclidean(u,v)

def canberra(u,v):
    return distance.canberra(u,v)

def hamming(u,v):
    return distance.hamming(u,v)

def unifrac(*args,**kwargs):
    raise NotImplementedError('Not implemented')
