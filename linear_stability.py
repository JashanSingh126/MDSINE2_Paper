'''Linear stability classes
'''
# load packages
import numpy as np
import numpy.random as npr
import logging
import copy
import math
import time
import pickle
import sys
import random
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import linear_stability
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.spatial.distance
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from scipy.stats import truncnorm, bernoulli
from numpy import linalg as LA
from itertools import combinations
import random
import sys

import synthetic

def make_synthetic_stability_class(init_premade):
    syndata = synthetic.SyntheticData(init_premade=init_premade)

    return StabilityData(
            growth_rates=syndata.growth,
            interactions=syndata.A,
            species_names=syndata.asvs.names.order,
            interactions_obj=syndata.interactions)

class KeystonenessInstance:
    '''
    orig_ss (array)
        These are the steady-state values of the ASV `names` in the original
        configuration
    sub_ss (array)
        These are the steady-state values of the ASV `names` from the subsampled
        clusters
    '''

    def __init__(self, orig_ss, sub_ss, names):
        self.orig_ss = orig_ss
        self.sub_ss = sub_ss

        if orig_ss.shape != sub_ss.shape:
            print('orig_shape', orig_ss.shape)
            print('sub_shape', sub_ss.shape)
            sys.exit()

        self.names = names

        self.distance = self.calc_distance('euclidean')

    def calc_distance(self, metric='euclidean'):

        if metric == 'euclidean':
            return np.sqrt(np.sum( (self.orig_ss - self.sub_ss)**2))
        else:
            raise ValueError('Not implemented')



class StabilityData(object):

    '''This class is used to perform all the computation pertaining linear stability and keystoness
    analysis.
    '''

    def __init__(self, growth_rates, interactions, species_names, interactions_obj=None):
        self.growth_rates = growth_rates
        self.interactions = interactions
        self.species_names = species_names
        self.interactions_obj = interactions_obj

        self.combination_dict = {}
        self.keystoneness_dict = {}

    def get_inferred_growth_and_interactions(self, fname, separator):
         self.growth_rates = self.read_growth_rates_from_mdsine1_parameters_file(fname,separator)
         self.interactions = self.read_interactions_from_mdsine1_parameters_file(fname,separator)
         self.species_names = self.read_species_names_from_mdsine1_parameters_file(fname,separator)
         return self

    def get_combinations(self, n):
        if n not in self.combination_dict:
            self.combination_dict[n] = self.calculate_combinations_of_size_n(n)
        return self.combination_dict[n]


    def get_steady_state(self):
        self.steady_state = self.calculate_steady_state()
        return self

    def get_jacobian(self):
        self.jacobian = self.calculate_jacobian()
        return self

    def get_eigenvalues(self):
        self.eigenvalues = self.calculate_eigenvalues()
        return self

    def get_combinations_of_size_n(self, num_comb):
        self.combinations_of_size_n = self.calculate_combinations_of_size_n(num_comb)
        return self

    def plot_eigenvalues(self):
        self.argand_plot()

    def read_interactions_from_mdsine1_parameters_file(self,fname,separator):
        '''Make the interaction matrix by reading from paramater file written by mdsine1.
        '''
        if separator is None:
            separator = ","
        df_pars = pd.read_csv(fname,sep=separator)
        df_A = df_pars.loc[df_pars['parameter_type'] == "interaction"] # extract interactions
        arr = df_A.values
        rows, row_pos = np.unique(arr[:, 1], return_inverse=True)
        cols, col_pos = np.unique(arr[:, 2], return_inverse=True)
        np_A = np.zeros((len(rows), len(cols)))
        np_A[row_pos, col_pos] = arr[:, 3]
        A = np_A
        return A

    def read_growth_rates_from_mdsine1_parameters_file(self,fname,separator):
        '''Make the growth rates matrix by reading from paramater file written by mdsine1.
        '''
        if separator is None:
            separator = ","
        df_pars = pd.read_csv(fname,sep=separator)
        df_r = df_pars.loc[df_pars['parameter_type'] == "growth_rate"] # extract growth rates
        arr0 = df_r.values
        rows0, row_pos0 = np.unique(arr0[:, 2], return_inverse=True)
        np_r = np.zeros((len(rows0), 1))
        np_r[row_pos0, 0] = arr0[:, 3]
        r = np_r
        return r

    def read_species_names_from_mdsine1_parameters_file(self,fname,separator):
        '''Make the interaction matrix by reading from paramater file written by mdsine1.
        '''
        if separator is None:
            separator = ","
        df_pars = pd.read_csv(fname,sep=separator)
        df_r = df_pars.loc[df_pars['parameter_type'] == "growth_rate"] # extract growth rates
        arr0 = df_r.values
        rows0, row_pos0 = np.unique(arr0[:, 2], return_inverse=True)
        return rows0
    #
    def calculate_steady_state(self):
        '''Calculate steadys states corresponding to a certain profile
        '''
        A = np.asmatrix(self.interactions)
        r = np.asmatrix(self.growth_rates).reshape(-1,1)
        ss = - np.linalg.pinv(A)*r
        return ss
    #
    def calculate_jacobian(self):
        ss = self.steady_state
        A = self.interactions
        r = self.growth_rates
        J = np.zeros(A.shape)
        print(J.shape[1])
        # nested for loop to crated the Jacobian
        for i in range(1,J.shape[1]-1):
            for j in range(1, J.shape[1]-1):
                if i == j:
                    sum_ij=0
                    for k in range(1,J.shape[1]-1):
                        sum_ij=sum_ij+A[i,k]*ss[k]
                        J[i,j]=r[i]+A[i,j]*ss[i]+sum_ij
                else:
                    J[i,j]=A[i,j]*ss[i]

        return(J)

    def calculate_eigenvalues(self):
        J = self.jacobian
        w, v = LA.eig(J)
        return w

    def argand_plot(self):
        import matplotlib.pyplot as plt
        a = self.eigenvalues
        for x in range(len(a)):
            plt.plot([0,a[x].real],[0,a[x].imag],'ro-',label='python')
        #limit=np.max(np.ceil(np.absolute(a))) # set limits for axis
        #plt.xlim((-limit,limit))
        #plt.ylim((-limit,limit))
        plt.ylabel('Imaginary')
        plt.xlabel('Real')
        plt.show()
        return plt

    def calculate_combinations_of_size_n(self,num_comb):
        grt=self.growth_rates
        nspecies=grt.shape[0]
        pvec=np.arange(1,nspecies+1)-1
        csize = num_comb # size of all the combos to be determined
        comb = list(combinations(pvec,csize))
        return comb

    def calculate_cluster_combinations_of_size_n(self,n):
        if self.interactions_obj is None:
            raise ValueError('No clustering')

        n_clusters=len(self.interactions_obj.clustering)
        pvec=np.arange(1,n_clusters+1)-1
        csize = n # size of all the combos to be determined
        comb = list(combinations(pvec,csize))
        return comb

    def calculate_keystoneness(self, n):
        if self.self.interactions_obj is None:
            raise ValueError('No clustering')

        combs = self.calculate_cluster_combinations_of_size_n(n)
        print('combs',combs)

        orig_ss_master = np.asarray(self.calculate_steady_state()).flatten()

        print('orig_ss_master', orig_ss_master)
        #sys.exit()

        for i, comb in enumerate(combs):
            print('\n\ncomb',comb)
            comb = list(comb)

            growth, A, names = self.subsampled_stability_w_clusters(on_cidxs=comb)

            sub_tmp = linear_stability.StabilityData(growth,A,names)
            # calculate the steady state - if negative toss it otherwise move to get the eigenvalues
            sub_tmp = sub_tmp.get_steady_state()

            # keep going if no negative entries in the steady state
            a = sub_tmp.steady_state
            neg_a=a[a<0]
            if neg_a.size == 0:
                sub_tmp = sub_tmp.get_jacobian()
                sub_tmp = sub_tmp.get_eigenvalues()
                b = sub_tmp.eigenvalues
                pos_b = b[b>0]
                # keep going if no positive eigenvalues
                if pos_b.size == 0:
                    sub_ss = np.asarray(sub_tmp.steady_state).flatten()
                    print('comb good')
                    print('ss',sub_ss)
                    print('names',names)
                    names = sub_tmp.species_names
                    orig_ss = []

                    for name in names:
                        idx = self.interactions_obj.clustering.items.names.index[name]
                        orig_ss.append(orig_ss_master[idx])

                    orig_ss = np.asarray(orig_ss)
                    print('orig_ss', orig_ss)
                    sub_ss = np.asarray(sub_ss)

                    self.keystoneness_dict[tuple(comb)] = KeystonenessInstance(
                        orig_ss=orig_ss,
                        sub_ss=sub_ss,
                        names=names)
                    print('distance', self.keystoneness_dict[tuple(comb)].distance)




    def perform_combinatorial_cluster_stability(self, output_folder, n):
        if self.interactions_obj is None:
            raise ValueError('No clustering')

        combs = self.calculate_cluster_combinations_of_size_n(n)

        for i, comb in enumerate(combs):
            comb = list(comb)

            growth, A, names = self.subsampled_stability_w_clusters(on_cidxs=comb)

            sub_tmp = linear_stability.StabilityData(growth,A,names)
            # calculate the steady state - if negative toss it otherwise move to get the eigenvalues
            sub_tmp = sub_tmp.get_steady_state()
            # keep going if no negative entries in the steady state
            a = sub_tmp.steady_state
            neg_a=a[a<0]
            if neg_a.size == 0:
                sub_tmp = sub_tmp.get_jacobian()
                sub_tmp = sub_tmp.get_eigenvalues()
                b = sub_tmp.eigenvalues
                pos_b = b[b>0]
                # keep going if no positive eigenvalues
                if pos_b.size == 0:
                    print('saving comb', comb)
                    c0 = sub_tmp.species_names
                    c1 = sub_tmp.steady_state
                    c2 = sub_tmp.growth_rates
                    c3 = sub_tmp.interactions
                    c4 = np.concatenate((c1,c2,c3), axis=1)
                    c5 = pd.DataFrame(c4)
                    c5.index = c0
                    outfile = "{0}/combinatorial_{1}.csv".format(output_folder,i)
                    export_csv = c5.to_csv(outfile,header=False)
                else:
                    print('Not saving comb:', comb)
            else:
                print('Not saving comb:', comb)
        return c5

    def perform_combinatorial_stability(self,output_folder, n):
        comb = self.get_combinations(n)
        gr = self.growth_rates
        itr = self.interactions
        sn = self.species_names
        for i in range(len(comb)):
            i_ts = list(comb[i])
            gr_i = gr[i_ts]
            itr_i = itr[np.ix_(i_ts,i_ts)]
            sn_i = sn[i_ts]
            sub_tmp = linear_stability.StabilityData(gr_i,itr_i,sn_i)
            # calculate the steady state - if negative toss it otherwise move to get the eigenvalues
            sub_tmp = sub_tmp.get_steady_state()
            # keep going if no negative entries in the steady state
            a = sub_tmp.steady_state
            neg_a=a[a<0]
            if neg_a.size == 0:
                sub_tmp = sub_tmp.get_jacobian()
                sub_tmp = sub_tmp.get_eigenvalues()
                b = sub_tmp.eigenvalues
                pos_b = b[b>0]
                # keep going if no positive eigenvalues
                if pos_b.size == 0:
                    c0 = sub_tmp.species_names
                    c1 = sub_tmp.steady_state
                    c2 = sub_tmp.growth_rates
                    c3 = sub_tmp.interactions
                    c4 = np.concatenate((c1,c2,c3), axis=1)
                    c5 = pd.DataFrame(c4)
                    c5.index = c0
                    outfile = "{0}/combinatorial_{1}.csv".format(output_folder,i)
                    export_csv = c5.to_csv(outfile,header=False)
        return c5

    def subsampled_stability_w_clusters(self, on_cids=None, on_cidxs=None):
        '''This function creates an ASV-ASV matrix with the specified Cluster IDs
        Turned off. You can either specify the index of the cluster (`on_cidxs`)
        or the cluster IDs (`on_cids`) that you want to turn off.

        ---------
        args
        ---------
        on_cids (list(int), int, Optional)
            - A list of cluster IDs (or a single cluster ID) you want to turn on interactions to
            - If this is specified you do not need to specify `on_cidxs`
        on_cidxs (list(int), int, Optional)
            - A list of cluster indices (or a single cluster index) you want to turn on interactions to
            - If this is specified you do not need to specify `on_cids`
        '''
        # Type checking
        if self.interactions_obj is None:
            raise ValueError('No clustering object specified')

        if on_cids is None and on_cidxs is None:
            raise ValueError('Either `on_cids` or `off_cidxs` must be specified (not None)')
        if on_cids is not None and on_cidxs is not None:
            raise ValueError('Only one of `on_cids` or `off_cidxs` must be specified, not both')
        if on_cidxs is not None:
            # convert the cidxs into cids
            on_cids = []
            for idx in on_cidxs:
                #if type(idx) != int:
                #    print(type(idx))
                #    raise ValueError('Each element in `off_cidxs` must be an int')
                if idx >= len(self.interactions_obj.clustering):
                    raise ValueError('`{}` is out of range. There are only {} clusters'.format(
                        idx, self.interactions_obj.clustering.n_clusters))
                on_cids.append(self.interactions_obj.clustering.order[idx])

        if type(on_cids) == np.ndarray:
            on_cids = list(on_cids)
        if type(on_cids) == int:
            on_cids = [on_cids]
        if type(on_cids) != list:
            raise ValueError('`on_cids` ({}) must either be array_like or an int')
        for ele in on_cids:
            if type(ele) != int:
                raise ValueError('All elements in on_cids must be ints (ids)')
            if ele not in self.interactions_obj.clustering.clusters:
                raise ValueError('`{}` is not a cluster id'.format(ele))

        n_asvs = len(self.interactions_obj.clustering.items)
        growth = np.asarray(self.growth_rates).flatten()
        self_interactions = np.asarray(np.diag(self.interactions)).flatten()

        if len(growth) != n_asvs:
            raise ValueError('The length of growth ({}) is not the same as the number ' \
                'of ASVs ({})'.format(len(growth), n_asvs))
        if len(self_interactions) != n_asvs:
            raise ValueError('The length of self_interactions ({}) is not the same as the number ' \
                'of ASVs ({})'.format(len(self_interactions), n_asvs))

        # convert to a set for efficient lookup
        on_cids = set(on_cids)

        # build matrix while setting all interactions to and from the specified clusters
        # to zero
        A = np.zeros(shape=(n_asvs,n_asvs), dtype=float)
        # tid == target cluster id
        for tid in self.interactions_obj.clustering.order:
            # sid == source cluster id
            for sid in self.cinteractions_obj.lustering.order:
                if tid == sid and not self.interactions_obj.include_self_interactions:
                    continue

                # If the source or target id are not in on_cids, skip
                if tid not in on_cids or sid not in on_cids:
                    continue

                # If it is a negative indicator, skip
                if not self.interactions_obj.[tid][sid].indicator:
                    continue


                for toidx in self.interactions_obj.clustering.clusters[tid].members:
                    for soidx in self.interactions_obj.clustering.clusters[sid].members:
                        A[toidx,soidx] = self.interactions_obj[tid][sid].value

        # Set self interactions to the diagonal entries of the matrix
        for i in range(n_asvs):
            A[i,i] = self_interactions[i]


        # take out ASVs that are in the clusters to delete
        idxs_to_delete = []
        for cid in self.interactions_obj.clustering.order:
            if cid in on_cids:
                continue
            idxs_to_delete += list(self.interactions_obj.clustering.clusters[cid].members)

        growth = np.delete(growth, idxs_to_delete).reshape(-1,1)
        A = np.delete(A, idxs_to_delete, axis=0)
        A = np.delete(A, idxs_to_delete, axis=1)

        names = np.delete(self.interactions_obj.clustering.items.names.order, idxs_to_delete)

        return growth, A, names


    def perform_combinatorial_stability_w_invasion(self,output_folder,inv_name, inv_density, num_states, n):
        comb = self.get_combinations(n)
        gr = self.growth_rates
        itr = self.interactions
        sn = self.species_names

        tmp_list_1 = self.species_names == inv_name
        tmp_list_res_1 = list(filter(lambda i: tmp_list_1[i], range(len(tmp_list_1))))
        if num_states !=0:
            rand_item = random.sample(comb,num_states)
            comb_fil = np.asarray(rand_item)
        else:
            comb_fil = np.asarray(comb)
        tmp_2= np.zeros(comb_fil.shape)
        tmp_2[comb_fil==tmp_list_res_1[0]] = 1
        tmp_list_2 = np.sum(tmp_2, axis=1)>0
        tmp_list_res_2 = list(filter(lambda i: tmp_list_2[i], range(len(tmp_list_2))))
        comb_fil_2 = np.delete(comb_fil, tmp_list_res_2, axis=0)

        # remove the rows and cols corresponding to inv_name
        for i in range(0,len(comb_fil_2)-1):
        #for i in range(0,2):
            i_ts = comb_fil_2[i]
            gr_i = gr[i_ts]
            itr_i = itr[np.ix_(i_ts,i_ts)]
            sn_i = sn[i_ts]
            sub_tmp = linear_stability.StabilityData(gr_i,itr_i,sn_i)
            # calculate the steady state - if negative toss it otherwise move to get the eigenvalues
            sub_tmp = sub_tmp.get_steady_state()
            # keep going if no negative entries in the steady state
            a = sub_tmp.steady_state
            neg_a=a[a<0]
            if neg_a.size == 0:
                sub_tmp = sub_tmp.get_jacobian()
                sub_tmp = sub_tmp.get_eigenvalues()
                b = sub_tmp.eigenvalues
                pos_b = b[b>0]
                # keep going if no positive eigenvalues
                if pos_b.size == 0:
                    index_to_keep = np.append(i_ts,tmp_list_res_1[0])
                    c0 = sn[index_to_keep]
                    #print(c0)
                    c1 = sub_tmp.steady_state
                    c1 = np.append(np.asarray(c1),inv_density)
                    c1 = np.transpose(np.array([c1]))
                    c2 = np.asarray(gr[index_to_keep])
                    c3 = itr[np.ix_(index_to_keep,index_to_keep)]
                    c4 = np.concatenate((c1,c2,c3), axis=1)
                    c5 = pd.DataFrame(c4)
                    c5.index = c0
                    outfile = "{0}/combinatorial_with_invasion{1}.csv".format(output_folder,i)
                    export_csv = c5.to_csv(outfile,header=False)
          #return(c5)
