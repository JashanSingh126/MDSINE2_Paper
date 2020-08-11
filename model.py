'''Make the Dynamics
'''
import numpy as np
import time
import math

import pylab as pl

class gLVDynamicsSingleClustering(pl.BaseDynamics):
    '''Discretized Generalized Lotka-Voltera Dynamics with 
    clustered interactions and perturbations. This class provides functionality
    to forward simulate the dynamics:

    If log dynamics:
        log(x_{k+1}) = log(x_k) + \Delta_k * ( 
                a1*(1 + \gamma) + 
                x_k * a2 + 
                sum_{c_i != c_j} b_{c_i, c_j} x_{k,j})
    Else regular dynamics
        x_{k+1} = x_k + \Delta_k * x_k ( 
                a1*(1 + \gamma) + 
                x_k * a2 + 
                sum_{c_i != c_j} b_{c_i, c_j} x_{k,j})

    If you want to forward simulate, then pass this dynamics object into 
    `pylab.dynamics.integrate`. 
    
    NOTE
    ----
    THE CLUSTERING FOR THE INTERACTIONS AND THE PERTURBATIONS ARE THE SAME.
    THIS ASSUMES THAT THERE ARE NO OVERLAPPING PERTURBATIONS

    Attributes
    ----------
    growth : np.ndarray((n,)), pylab.variables.Variable((n,))
        This is the growth of the dynamics for each ASV
    self_interactions : np.ndarray((n,)), pylab.variables.Variable((n,))
        This is the self interactions for each of the ASVs
    interactions : np.ndarray((n,n)), pylab.cluster.Interactions, None
        If this is an array, it is assumed this is the ASV-ASV interactions (shape=(n,n)).
        The diagonal of this matrix WILL BE overriden with the `self_interactions`.
        Otherwise this can be initialized as the interactions object in pylab.cluster.
        If None then we assume that there are no interactions
    clustering : pylab.cluster.Clustering, Optional
        This is the clustering object that is used for the perturbations and the interactions.
        This is optional because we do not actually need this for the integration.
    perturbations : list(pylab.contrib.ClusterPerturbation), list((numeric, numeric, np.ndarray((n,)))), None
        These are the perturbations. Must be a list of Cluster Perturbation objects or a list of
        3-tuples: (start, end, effect). If None then we assume there are no perturbations

    Parameters
    ----------
    log_dynamics : bool
        If True, we log integrate the dynamics
    perturbations_additive : bool, optional
        If True, then we set the perturbation effect as dditive instread of multiplicative to
        the growth.

    kwargs
    ------
    asvs : pylab.base.ASVSet
        Holds all of the ASVs
    

    See also
    --------
    pylab.dynamics.integrate
    pylab.dynamics.BaseDynamics
    '''

    def __init__(self, log_dynamics, perturbations_additive, sim_max=None, 
        start_day=0,**kwargs):
        pl.BaseDynamics.__init__(self, **kwargs)

        self.growth = None
        self.self_interactions = None
        self.interactions = None
        self.clustering = None
        self.perturbations = None
        self.log_dynamics = log_dynamics
        self.perturbations_additive = perturbations_additive
        self.sim_max = sim_max
        self.start_day = start_day

    def stability(self):
        if self.growth is None or self.self_interactions is None or \
            self.interactions is None:
            raise ValueError('Cannot calculate stability')
        r = self.growth.value.reshape(-1,1)
        A = np.diag(-self.self_interactions.value)
        A += self.interactions.get_datalevel_value_matrix(
            set_neg_indicators_to_nan=False)
        return - np.linalg.pinv(A) @ r

    def init_integration(self):
        '''Check that everything is initialized correctly. 
        Convert to numpy arrays if necessary
        '''
        # Type checking and setting interior values for faster integration
        # growth
        if pl.isVariable(self.growth):
            self._growth = self.growth.value
        elif pl.isarray(self.growth):
            self._growth = self.growth
        else:
            raise TypeError('`growth` ({}) must be a pylab.varaibles.Variable or a ' \
                'numpy.ndarray'.format(type(self.growth)))
        self._growth = np.asarray(self._growth)
        # if len(self._growth) != len(self.asvs):
        #     raise ValueError('`growth` ({}) must be ({}) long'.format(
        #         len(self._growth), len(self.asvs)))

        # Self-interactions
        if pl.isVariable(self.self_interactions):
            self._self_interactions = self.self_interactions.value
        elif pl.isarray(self.self_interactions):
            self._self_interactions = self.self_interactions
        else:
            raise TypeError('`self_interactions` ({}) must be a pylab.varaibles.Variable or a ' \
                'numpy.ndarray'.format(type(self.self_interactions)))
        self._self_interactions = np.asarray(self._self_interactions)
        # if len(self._self_interactions) != len(self.asvs):
        #     raise ValueError('`self_interactions` ({}) must be ({}) long'.format(
        #         len(self._self_interactions), len(self.asvs)))

        # Interactions
        if self.interactions is not None:
            if pl.isinteractions(self.interactions):
                self._interactions = self.interactions.get_datalevel_value_matrix(
                    set_neg_indicators_to_nan=False)
            elif pl.isarray(self.interactions):
                self._interactions = self.interactions
            else:
                raise TypeError('`interactions` ({}) must be an array or pylab.cluster.Interactions' \
                    ''.format(type(self.interactions)))
            self._interactions = np.asarray(self._interactions)
            self._interactions = np.nan_to_num(self._interactions)
            if len(self._interactions.shape) != 2:
                raise ValueError('`interactions` ({}) must be a 2 dimensional array'.format(
                    len(self._interactions.shape)))
            if self._interactions.shape[0] != self._interactions.shape[1]:
                raise ValueError('`interactions` ({}) must be a square matrix'.format(
                    self._interactions.shape))
            # if self._interactions.shape[0] != len(self.asvs):
            #     raise ValueError('`interactions` ({}) must be ({}) elements long'.format(
            #         self._interactions.shape[0], len(self.asvs)))
        else:
            self._interactions = None
        
        # Perturbations
        if self.perturbations is not None:
            if not pl.isarray(self.perturbations):
                raise TypeError('`perturbations ({}) must be an array'.format(type(self.perturbations)))
            self._perturbations = []
            for perturbation in self.perturbations:
                if pl.isclusterperturbation(perturbation):
                    self._perturbations.append((
                        perturbation.start, perturbation.end, perturbation.item_array()))
                elif pl.istuple(perturbation):
                    if len(perturbation) != 3:
                        raise ValueError('`perturbation` ({}) if a tuple must be 3 elements long'.format(
                            len(perturbation)))
                    start, end, effect = perturbation
                    if not np.all(pl.itercheck([start,end], pl.isnumeric)):
                        raise TypeError('`start` ({}) and `end` ({}) must be numerics'.format(
                            type(start), type(end)))
                    if start >= end:
                        raise ValueError('`start` ({}) must be less than `end` ({})'.format(start, end))
                    if not pl.isarray(effect):
                        raise TypeError('`effect` ({}) must be an array'.format(type(effect)))
                    effect = np.asarray(effect)
                    effect = np.nan_to_num(effect)
                    # if len(effect) != len(self.asvs):
                    #     raise ValueError('the length of `effect` ({}) must be ({})'.format( 
                    #         len(effect), len(self.asvs)))
                    start -= self.start_day
                    end -= self.start_day
                    self._perturbations.append((start,end,effect))

                else:
                    raise TypeError('`perturbation` ({}) must either be a tuple or a ' \
                        'pylab.contrib.ClusterPerturbation'.format(type(perturbation)))
        else:
            self._perturbations = None
        

        # Everything checked and set
        self._default_growth = self._growth.reshape(-1,1)
        for i in range(len(self._self_interactions)): #range(len(self.asvs)):
            self._interactions[i,i] = -self._self_interactions[i]
        
        temp = []
        if self._perturbations is not None:
            temp = []
            for start,end,effect in self._perturbations:
                if self.perturbations_additive:
                    temp.append(
                        (start, end, effect.reshape(-1,1)))
                else:
                    temp.append(
                        (start, end,
                        self._default_growth * (1 + effect.reshape(-1,1))))
            
            # Sort the perturbations by their start time 
            self._perturbations = []
            while len(temp) > 0:
                idx = None
                prev = float('inf')
                for i,(start, _, _) in enumerate(temp):
                    if start < prev:
                        idx = i
                        prev = start
                if idx is None:
                    raise ValueError('catbutt')
                self._perturbations.append(temp.pop(idx))
            self._curr_pert_start, self._curr_pert_end, _ = self._perturbations[0]
            self._in_pert = False
            self._pert_idx = 0
            
    def integrate_single_timestep(self, x, t, dt):
        '''Integrate over a single step

        Parameters
        ----------
        x : np.ndarray((n,1))
            This is the abundance as a column vector for each ASV
        t : numeric
            This is the time point we are integrating to
        dt : numeric
            This is the amount of time from the previous time point we
            are integrating from
        '''
        growth = self._default_growth
        perts = None

        # Adjust the growth depending if we are in a perturbation or not
        if self._perturbations is not None:
            if not self._in_pert:
                # If we were not in a perturabtion, chcek to see if we should be
                if t >= self._curr_pert_start:
                    # switch the the perturbation to on.
                    # Get an new growth if necessary
                    self._in_pert = True
                    if self.perturbations_additive:
                        perts = self._perturbations[self._pert_idx][2]
                    else:
                        growth = self._perturbations[self._pert_idx][2]
            else:
                # If we are currently in the perturbation, check to see if it will change
                if t >= self._curr_pert_end:
                    # switch to being off in go to the next perturbation if possible
                    self._in_pert = False
                    self._pert_idx += 1
                    if self._pert_idx == len(self._perturbations):
                        # No more perturbations
                        self._curr_pert_start = float('inf')
                    else:
                        self._curr_pert_start, self._curr_pert_end, _ = \
                            self._perturbations[self._pert_idx]
                    
                else:
                    # we are still in the perturbation
                    if self.perturbations_additive:
                        perts = self._perturbations[self._pert_idx][2]
                    else:
                        growth = self._perturbations[self._pert_idx][2]
        
        if self.perturbations_additive and perts is None:
            perts = 0

        # Integrate
        if self.log_dynamics:
            if self.perturbations_additive:
                ret = np.exp(np.log(x) + (growth + self._interactions @ x + perts) * dt).ravel()
            else:
                ret = np.exp(np.log(x) + (growth + self._interactions @ x) * dt).ravel()
        else:
            if self.perturbations_additive:
                ret = (x + x * (growth + self._interactions @ x + perts) * dt).ravel()
            else:
                ret = (x + x * (growth + self._interactions @ x) * dt).ravel()
        
        if self.sim_max is not None:
            ret[ret >= self.sim_max] = self.sim_max
        return ret

    def finish_integration(self):
        '''This is the function that
        '''
        self._default_growth = None
        self._growth = None
        self._interactions = None
        self._perturbations = None
        self._curr_pert_start = None
        self._curr_pert_end = None
        self._in_pert = None
        self._pert_idx = None
        

class HeteroscedasticGlobal(pl.BaseProcessVariance):

    def __init__(self, *args, **kwargs):
        pl.BaseProcessVariance.__init__(self, *args, **kwargs)

        self.v1 = None
        self.v2 = None
        self.c_m = None

    def set_parameters(self, v1, v2, c_m):
        self.v1 = v1
        self.v2 = v2 
        self.c_m = c_m

    def init_integration(self):
        if self.v1 is None:
            raise TypeError('`v1` ({}) needs to be initialized'.format(
                type(self.v1)))
        if self.v2 is None:
            raise TypeError('`v2` ({}) needs to be initialized'.format(
                type(self.v2)))
        if self.c_m is None:
            raise TypeError('`c_m` ({}) needs to be initialized'.format(
                type(self.c_m)))
        if not pl.isVariable(self.v1):
            self._v1 = pl.Variable(value=self.v1)
        else:
            self._v1 = self.v1
        if not pl.isVariable(self.v2):
            self._v2 = pl.Variable(value=self.v2)
        else:
            self._v2 = self.v2
        if not pl.isVariable(self.c_m):
            self._c_m = pl.Variable(value=self.c_m)
        else:
            self._c_m = self.c_m
        
        v = np.asarray([self._v1.value, self._v2.value, self._c_m.value])
        if not np.all(pl.itercheck(v, pl.isnumeric)):
            raise TypeError('`v1` ({}), `v2` ({}), `c_m` ({}) must be numerics'.format( 
                type(self._v1.value), type(self._v2.value), type(self._c_m.value)))
        if np.any(v < 0):
            raise ValueError('`v1` ({}), `v2` ({}), `c_m` ({}) must be >= 0'.format( 
                self._v1.value, self._v2.value, self._c_m.value))

        self._v1 = self._v1.value
        self._v2 = self._v2.value
        self._c_m = self._c_m.value

        self._offset = self._v2 * (self._c_m ** 2)

    # @profile
    def integrate_single_timestep(self, x, t, dt):
        return pl.random.truncnormal.sample_vec(
            mean=x, 
            std=np.sqrt((self._v1*x**2 + self._offset)*dt), 
            low=0, high=float('inf'))
    
    def finish_integration(self):
        self._v1 = None
        self._v2 = None
        self._c_m = None


class HeteroscedasticPerASV(HeteroscedasticGlobal):

    def init_integration(self):
        if self.v1 is None:
            raise TypeError('`v1` ({}) needs to be initialized'.format(
                type(self.v1)))
        if self.v2 is None:
            raise TypeError('`v2` ({}) needs to be initialized'.format(
                type(self.v2)))
        if self.c_m is None:
            raise TypeError('`c_m` ({}) needs to be initialized'.format(
                type(self.c_m)))
        if not pl.isVariable(self.v1):
            self._v1 = self.v1
        else:
            self._v1 = self.v1.value
        if not pl.isVariable(self.v1):
            self._v2 = self.v2
        else:
            self._v2 = self.v2.value
        if not pl.isVariable(self.c_m):
            self._c_m = self.c_m
        else:
            self._c_m = self.c_m.value

        v = np.asarray([self._v1, self._v2])
        if not np.all(pl.itercheck(v, pl.isarray)):
            raise TypeError('`v1` ({}), `v2` ({}) must be arrays'.format( 
                type(self._v1), type(self._v2)))
        if len(self._v1) != len(self.asvs) or len(self._v2) != len(self.asvs):
            raise ValueError('`v1` ({}) and `v2` ({}) must be length ({})'.format(
                len(self._v1), len(self._v2), len(self.asvs)))
        if not pl.isnumeric(self._c_m):
            raise ValueError('`c_m` ({}) must be numeric'.format(type(self._c_m)))
        if np.any(v < 0) or self._c_m < 0:
            raise ValueError('`v1` ({}), `v2` ({}), `c_m` ({}) must be >= 0'.format( 
                self._v1, self._v2, self._c_m))

        self._v1 = self._v1.ravel()
        self._v2 = self._v2.ravel()
        self._offset = self._v2 * (self._c_m ** 2)


class MultiplicativeGlobal(pl.BaseProcessVariance):
    '''This is multiplicative noise used in a lognormal model
    '''
    def __init__(self, *args, **kwargs):
        pl.BaseProcessVariance.__init__(self, *args, **kwargs)

        self.value = None
    
    def set_parameters(self, value):
        self.value = value

    def init_integration(self):
        if self.value is None:
            raise ValueError('`value` needs to be initialized (it is None).')
        elif pl.isVariable(self.value):
            self._value = self.value.value
        else:
            self._value = self.value
        if not pl.isnumeric(self._value):
            raise TypeError('`value` ({}) must be a numeric'.format(type(self.value)))

    def integrate_single_timestep(self, x, t, dt):
        std = np.sqrt(dt * self._value)
        return np.exp(pl.random.normal.sample(mean=np.log(x), std=std))

    def finish_integration(self):
        self._value = None



