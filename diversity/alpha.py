'''Alpha diversity measures

Defined measures:
    * berger_parker_dominance
    * brillouin_index
    * dominance
    * doubles
    * singles
    * enspie
    * etsy_ci
    * goods_coverage
    * heip_evenness
    * kempton_taylor_q_index
    * margalef_richness
    * mcintosh_dominance
    * mcintosh_evenness
    * menhinick_richness
    * observed_asvs
    * pielou_evenness
    * robbins
    * shannon_entropy
    * simpson_index
    * simpson_evenness
    * strong_dominance
    * ace
    * chao1
'''
import numpy as np
import scipy.special

def entropy(counts):
    '''Calculate the entropy
    
    Entropy is defined as
        E = - \sum_i (b_i * \log(b_i))
    where
        b_i is the relative abundance of the ith ASV
    
    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    counts = _validate_counts(counts)
    rel = counts[counts>0]
    rel = rel / np.sum(rel)

    a = rel * np.log(rel)
    a = -np.sum(a)
    return a

def normalized_entropy(counts):
    '''Calculate the normailized entropy
    
    Entropy is defined as
        E = - \sum_i (b_i * \log_n(b_i))
    where
        b_i is the relative abundance of the ith ASV
    
    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    counts = _validate_counts(counts)
    rel = counts[counts>0]
    rel = rel / np.sum(rel)

    a = rel * np.log(rel)
    a = -np.sum(a) / np.log(len(rel))
    return a

def berger_parker_dominance(counts):
    '''Caculate the Berger-Parker dominance.

    The dominance is defined as the fraction of the sample that belongs
    to the most abundant ASV:

                    max(counts)/sum(counts)

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    counts = _validate_counts(counts)
    return np.max(counts)/np.sum(counts)

def brillouin_index(counts):
    '''Caculate the Brillouin index of alpha diversity.

    The dominance is calculated as:

                    (ln(N!) - sum(ln(n_i)))/N

    N: total number of counts
    n_i: number of counts for the ith ASV

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    counts = _validate_counts(counts)
    nz = counts[counts.nonzero()]
    n = nz.sum()

    return (scipy.special.gammaln(n+1)-scipy.special.gammaln(nz+1).sum())/n

def dominance(counts):
    '''Caculates dominance.

    The dominance is calculated as:

                    sum(p_i^2)
    where `p_i` is the relative abundance of ASV i.

    Also defined as 1- Simpson's index and it ranges between 0 and 1

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    counts = _validate_counts(counts)
    return np.sum(np.square(counts/np.sum(counts)))

def doubles(counts):
    '''Counts the number of double occurances (doubletons)

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    int
    '''
    counts = _validate_counts(counts)
    return (counts == 2).sum()

def singles(counts):
    '''Counts the number of single occurances (singletons)

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    int
    '''
    counts = _validate_counts(counts)
    return (counts == 1).sum()

def enspie(counts):
    '''Calculate ENS_pie alpha diversity measure

    Equivalent to `1/dominance`.

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    return 1/dominance(counts)

def etsy_ci(counts):
    '''Calculate Esty's confidence interval

                F1/N +- sqrt(W)

    F1: number of singletons
    N: total number of counts
    W: (F1 * (N-F1) + 2*N*F2)/(N^3)
    F2: Number of doubletons

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    2-tuple
        - (lower_bound, upper_bound)
    '''
    counts = _validate_counts(counts)
    f1 = singles(counts)
    f2 = doubles(counts)
    n = counts.sum()
    z = 1.959963985
    W = (f1*(n-f1)+2*n*f2)/(n**3)

    return f1/n-z*np.sqrt(W), f1/n+z*np.sqrt(W)

def goods_coverage(counts):
    '''Caculate Good's coverage of counts

            1-F1/N
    F1: number of singletons
    N: total number of counts

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    counts = _validate_counts(counts)
    f1 = singles(counts)
    N = counts.sum()
    return 1-(f1/N)

def heip_evenness(counts):
    '''Calculates Heip's evenness measure.

            (e^H - 1)/(S-1)

    H: shannon-Wiener entropy counts (using log base e)
    S: number of distinct asvs in the sample (non-zero)

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    counts = _validate_counts(counts)
    return (np.exp(shannon_entropy(counts, base=np.e))-1)/(observed_asvs(counts)-1)

def kempton_taylor_q_index(counts, lower_quantile=0.25, upper_quantile=0.75):
    '''Caclulates the Kempton-Taylor Q index.

    Estimates the slope of the cumlative abundance curve in the interquartile
    range. By default, uses lower and upper quartiles rounding inwards.

    Index is defined in [1], implementation is based on the description
    given in the SDR-IV online manual [2].

    [1] Kempton, R. A. and Taylor, L. R. (1976) Models and statistics for
        species diversity. Nature, 262, 818-820.
    [2] http://www.pisces-conservation.com/sdrhelp/index.html

    Parameters
    ----------
    counts (array_like)
        - Vector of counts
    lower_quantile (float, Optional)
        - lower bound of the interquantile range
    lower_quantile (float, Optional)
        - upper bound of the interquantile range

    Returns
    -------
    double
    '''
    counts = _validate_counts_vector(counts)
    n = len(counts)
    lower = int(np.ceil(n*lower_quantile))
    upper = int(n*upper_quantile)
    sorted_counts = np.sort(counts)
    return (upper-lower)/np.log(sorted_counts[upper]/sorted_counts[lower])

def margalef_richness(counts):
    '''Calculates Margalef's richness index.

                (S-1)/(ln(N))

    S: number of unique, observed ASVs (non-zero ASVs)
    N: total number of counts

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    counts = _validate_counts(counts)
    return (observed_asvs(counts)-1)/(np.log(counts.sum()))

def mcintosh_dominance(counts):
    '''Calculates the McIntosh dominance index

            (N-sqrt(sum(n_i^2)))/(N-sqrt(N))

    N: total number of counts
    n_i: number of counts in the ith ASV


    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    counts = _validate_counts(counts)
    u = np.sqrt(np.sum(counts*counts))
    n = counts.sum()
    return (n-u)/(n-np.sqrt(n))

def mcintosh_evenness(counts):
    '''Caculates the McIntosh evenness measure

            (sqrt(sum(n_i^2)))/sqrt((N-S+1)^2 + S - 1)

    n_i: number of counts for ASV i
    N: total number of counts
    S: number of non-zero ASVs

    Based on the implementation in [1]

    [1] Heip & Engels (1974) Comparing Species Diversity and Evenness
        Indices. p 560.

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    counts = _validate_counts(counts)
    n = counts.sum()
    s = observed_counts(counts)
    return np.sqrt(np.sum(counts*counts))/np.sqrt((n-s+1)**2+s-1)

def menhinick_richness(counts):
    '''Caculates the Menhinick Richness

                S/sqrt(N)
    S: number of non-zero ASVs
    N: total number of counts

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    counts = _validate_counts(counts)
    return observed_counts(counts)/np.sqrt(counts.sum())

def observed_asvs(counts):
    '''Caclulate the number of distinct ASVs

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    counts = _validate_counts(counts)
    return np.sum(counts != 0)

def pielou_evenness(counts):
    '''Caculates Pielou's evenness

            H/(len(S))

    H: Shannon-Wiener entropy
    S: Number of non-zero ASVs

    Based on the implementation in [1].

    [1] https://en.wikipedia.org/wiki/Species_evenness

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    counts = _validate_counts(counts)
    return shannon_entropy(counts, base=np.e)/np.log(observed_asvs(counts))

def robbins(counts):
    '''Returns the probability of the getting an unobserved outcome.

                F1/(N+1)

    F1: number of singletons
    N: total number of counts

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    counts = _validate_counts(counts)
    return singles(counts)/np.sum(counts)

def shannon_entropy(counts, base=2):
    '''Calculates the Shannon entropy


    Based on the description given in the SDR-IV online manual [1] except that
    the default logarithm base used here is 2 instead of `e`.

    [1] http://www.pisces-conservation.com/sdrhelp/index.html

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    counts = _validate_counts(counts)
    freqs = counts/counts.sum()
    nonzero_freqs = freqs[freqs.nonzero()]
    return -(nonzero_freqs*np.log(nonzero_freqs)).sum()/np.log(base)

def simpson_index(counts):
    '''Caculates the Simpson index. This is defined as 1-dominance

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    return 1-dominance(counts)

def simpson_evenness(counts):
    '''Calculates the simpson evenness measure.

                (1-D)/(S)

    D: dominance
    S_obs: Number of non-zero ASVs

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    return enspie(counts)/observed_asvs(counts)

def strong_dominance(counts):
    '''Calculates Strong's dominance

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double
    '''
    counts = _validate_counts(counts)
    n = counts.sum()
    s = observed_asvs(counts)
    i = np.arange(1, len(counts) + 1)
    sorted_sum = np.sort(counts)[::-1].cumsum()
    return (sorted_sum / n - (i / s)).max()

def ace(counts, rare_threshold=10):
    '''Calculate the ACE metric (Abundance-based Coverage Estimator).

    Parameters
    ----------
    counts (array_like)
        - Vector of counts

    Returns
    -------
    double

    
    Raises
    ------
    ValueError
        If every rare ASV is a singleton.

    Notes
    -----
    The implementation here is based on the description given in the EstimateS
    manual [1]. If no rare ASVs exist, returns the number of abundant ASVs. The default
    value of 10 for `rare_threshold` is based on [2]. If `counts` contains zeros,
    indicating ASVs which are known to exist in the environment but did not appear
    in the sample, they will be ignored for the purpose of calculating the number of rare ASVs.
    
    References
    ----------
    [1] http://viceroy.eeb.uconn.edu/estimates/
    [2] Chao, A., W.-H. Hwang, Y.-C. Chen, and C.-Y. Kuo. 2000. Estimating
        the number of shared species in two communities. Statistica Sinica
        10:227-246.
    '''
    def _asvs_rare(freq_counts, rare_threshold):
        '''Count number of rare ASVs.'''
        return freq_counts[1:rare_threshold + 1].sum()

    def _asvs_abundant(freq_counts, rare_threshold):
        '''Count number of abundant ASVs.'''
        return freq_counts[rare_threshold + 1:].sum()

    def _number_rare(freq_counts, rare_threshold, gamma=False):
        '''Return number of individuals in rare ASVs.
        `gamma=True` generates the `n_rare` used for the variation coefficient.
        '''
        n_rare = 0
        if gamma:
            for i, j in enumerate(freq_counts[:rare_threshold + 1]):
                n_rare = n_rare + (i * j) * (i - 1)
        else:
            for i, j in enumerate(freq_counts[:rare_threshold + 1]):
                n_rare = n_rare + (i * j)
        return n_rare


    counts = _validate_counts(counts)
    freq_counts = np.bincount(counts)
    s_rare = _asvs_rare(freq_counts, rare_threshold)
    singles = freq_counts[1]

    if singles > 0 and singles == s_rare:
        raise ValueError('The only rare ASVs are singletons, so the ACE ' \
            'metric is undefined. EstimateS suggests using bias-corrected Chao1 instead.')

    s_abun = _asvs_abundant(freq_counts, rare_threshold)
    if s_rare == 0:
        return s_abun

    n_rare = _number_rare(freq_counts, rare_threshold)
    c_ace = 1 - singles / n_rare

    top = s_rare * _number_rare(freq_counts, rare_threshold, gamma=True)
    bottom = c_ace * n_rare * (n_rare - 1)
    gamma_ace = (top / bottom) - 1

    if gamma_ace < 0:
        gamma_ace = 0

    return s_abun + (s_rare / c_ace) + ((singles / c_ace) * gamma_ace)

def chao1(counts, bias_corrected=True):
    '''Calculates chao1 richness estimator

    Based on the implementation in [1]

    [1] Chao, A. 1984. Non-parametric estimation of the number of classes in
        a population. Scandinavian Journal of Statistics 11, 265-270.

    Parameters
    ----------
    counts (array_like)
        - Vector of counts
    bias_corrected (bool, Optional)
        - If True, uses the bias corrected verison of the equation

    Returns
    -------
    double
    '''
    o = observed_asvs(counts)
    s = singles(counts)
    d = doubles(counts)

    if not bias_corrected and s and d:
        return o+s**2/(d*2)
    else:
        return o+s*(s-1)/(2*(d+1))

# Utility functions
def _validate_counts(counts, cast_as_ints=True):
    '''Checks dimensions, wraps as np array, and casts values as ints
    if necessary

    Parameters
    ----------
    counts (array_like)
        - 1D data
    cast_as_ints (bool, Optional)
        - If True, it will cast the counts array as an int
        - If False it will not cast

    Returns
    -------
    np.ndarray
    '''

    counts = np.asarray(counts)
    if cast_as_ints:
        counts = counts.astype(int, copy=False)

    if counts.ndim != 1:
        raise ValueError('counts ({}) must be a single dimension'.format(
            counts.shape))
    if np.any(counts < 0):
        raise ValueError('counts must not have any negative values')
    return counts