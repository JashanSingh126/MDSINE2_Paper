import pickle
import re
import numpy as np
import numba
import scipy
import scipy.sparse


# Constants
NAME_FORMATTER = '%(name)s'
ID_FORMATTER = '%(id)s'
INDEX_FORMATTER = '%(index)s'
STRAIN_FORMATTER = '%(strain)s'
SPECIES_FORMATTER = '%(species)s'
SPECIESX_FORMATTER = '%(speciesX)s'
GENUS_FORMATTER = '%(genus)s'
FAMILY_FORMATTER = '%(family)s'
CLASS_FORMATTER = '%(class)s'
ORDER_FORMATTER = '%(order)s'
PHYLUM_FORMATTER = '%(phylum)s'
KINGDOM_FORMATTER = '%(kingdom)s'
LCA_FORMATTER = '%(lca)s'

_TAXLEVELS = ['strain', 'species', 'genus', 'family', 'class', 'order', 'phylum', 'kingdom']
_TAXFORMATTERS = ['%(strain)s', '%(species)s', '%(genus)s', '%(family)s', '%(class)s', '%(order)s', '%(phylum)s', '%(kingdom)s']
_SPECIESX_SEARCH = re.compile('\%\(species[0-9]+\)s')

class Saveable:
    '''Implements baseline saving classes with pickle for classes
    '''
    def save(self, filename=None):
        '''Pickle the object

        Paramters
        ---------
        filename : str
            This is the location to store the file. Overrides the location if
            it is set using `pylab.base.Saveable.set_save_location`. If None
            it means that we are using the file location set in 
            set_location. 
        '''
        if filename is None:
            if not hasattr(self, '_save_loc'):
                raise TypeError('`filename` must be specified if you have not ' \
                    'set the save location')
            filename = self._save_loc
        
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(self, output, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        '''Unpickle the object

        Paramters
        ---------
        filename : str
            This is the location of the file to unpickle
        '''
        with open(filename, 'rb') as handle:
            b = pickle.load(handle)
        if type(b) != cls:
            raise TypeError('Type {} not loaded. Loaded a {} class'.format(
                cls, type(b)))
        return b

    def set_save_location(self, filename):
        '''Set the save location for the object
        '''
        if not isstr(filename):
            raise TypeError('`filename` ({}) must be a str'.format(type(filename)))
        self._save_loc = filename

def isbool(a):
    '''Checks if `a` is a bool

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a bool
    '''
    return a is not None and np.issubdtype(type(a), np.bool_)

def isint(a):
    '''Checks if `a` is an int

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is an int
    '''
    return a is not None and np.issubdtype(type(a), np.integer)

def isfloat(a):
    '''Checks if `a` is a float

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a float
    '''
    return a is not None and np.issubdtype(type(a), np.floating)

def iscomplex(a):
    '''Checks if `a` is a complex number

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a complex number
    '''
    return a is not None and np.issubdtype(type(a), np.complexfloating)

def isnumeric(a):
    '''Checks if `a` is a float or an int - (cannot be a bool)

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a float or an int.
    '''
    return a is not None and np.issubdtype(type(a), np.number)

def isarray(a):
    '''Checks if `a` is an array

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is an array
    '''
    return (type(a) == np.ndarray or type(a) == list or \
        scipy.sparse.issparse(a)) and a is not None

def isstr(a):
    '''Checks if `a` is a str

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a str
    '''
    return a is not None and type(a) == str

def istype(a):
    '''Checks if `a` is a Type object

    Example
    -------
    >>> istype(5)
    False
    >>> istype(float)
    True

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a tuple
    '''
    return type(a) == type

def istuple(a):
    '''Checks if `a` is a tuple object

    Example
    -------
    >>> istuple(5)
    False
    >>> istuple((5,))
    True

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a tuple
    '''
    return type(a) == tuple

def isdict(a):
    '''Checks if `a` is a dict object

    Example
    -------
    >>> isdict(5)
    False
    >>> isdict({5:2})
    True

    Parameters
    ----------
    a : any
        Instance we are checking

    Returns
    -------
    bool
        True if `a` is a dict
    '''
    return type(a) == dict

def itercheck(xs, f):
    '''Checks every element in xs with f and returns an array
    for each entry

    Parameters
    ----------
    xs : array_like(any)
        - A list of instances to check the type of
    f : callable
        - Type checking function
    Returns
    -------
    list(bool)
        Checks for each element in the `xs`
    '''
    return [f(x) for x in xs]

def microbename_formatter(format, microbe, microbes, lca=True):
    '''Format the label of an microbe. Specify the microbe by
    it's index in the microbeSet `microbes`

    Example:
        microbe is an microbe object at index 0 where
        microbe.genus = 'A'
        microbe.id = 1234532

        microbename_formatter(
            format='%(genus)s: %(index)s',
            microbe=1234532,
            microbes=microbes)
        >>> 'A: 0'

        microbename_formatter(
            format='%(genus)s: %(genus)s',
            microbe=1234532,
            microbes=microbes)
        >>> 'A: A'

        microbename_formatter(
            format='%(index)s',
            microbe=1234532,
            microbes=microbes)
        >>> '0'

        microbename_formatter(
            format='%(geNus)s: %(genus)s',
            microbe=1234532,
            microbes=microbes)
        >>> '%(geNus)s: A'

    Parameters
    ----------
    format : str
        This is the format for us to do the labels
        Formatting options:
            '%(name)s'
                Name of the Microbe (pylab.base.Microbe.name)
            '%(id)s'
                ID of the Microbe (pylab.base.Microbe.id)
            '%(index)s'
                The order that this appears in the MicrobeSet
            '%(species)s'
                `'species'` taxonomic classification of the Microbe
            '%(speciesX)s'
                `'species'` taxonomic classification of the Microbe for only up to the first 
                `X` spceified
            '%(genus)s'
                `'genus'` taxonomic classification of the Microbe
            '%(family)s'
                `'family'` taxonomic classification of the Microbe
            '%(class)s'
                `'class'` taxonomic classification of the Microbe
            '%(order)s'
                `'order'` taxonomic classification of the Microbe
            '%(phylum)s'
                `'phylum'` taxonomic classification of the Microbe
            '%(kingdom)s'
                `'kingdom'` taxonomic classification of the Microbe
            '%(lca)s'
                Least common ancestor. If species is 'NA', then it will go to family.
                It will keep travelling up the tree until it finds something not nan.
                Example:
                microbe is an Microbe object at index 0 where
                microbe.genus = 'nan'
                microbes.family = 'B'
                microbe.id = 1234532

    microbe : str, int, Microbe
        Either the microbe or an id for the microbe
    microbes : pylab.base.MicrobeSet
        Dataset containing all of the information for the microbes
    lca : bool
        If True and the specified taxonomic level is not specified (nan), then
        we substitute it with the least common ancestor up from the current level

    '''
    microbe = microbes[microbe]
    index = microbe.idx

    label = format.replace(NAME_FORMATTER, str(microbe.name))
    label = label.replace(ID_FORMATTER, str(microbe.id))
    label = label.replace(INDEX_FORMATTER,  str(index))

    
    # Replcate speciesX formatter
    X = _SPECIESX_SEARCH.search(format)
    if X is not None:
        while True:
            X = X[0]
            n = int(X.replace('%(species', '').replace(')s',''))
            try:
                a = '/'.join(microbe.get_taxonomy('species').split('/')[:n])
            except:
                a = 'nan'
            format = format.replace(X,a)
            X = _SPECIESX_SEARCH.search(format)
            if X is None:
                break
    
    for i in range(len(_TAXLEVELS)):
        taxlevel = _TAXLEVELS[i]
        fmt = _TAXFORMATTERS[i]
        try:
            label = label.replace(fmt, str(microbe.get_taxonomy(taxlevel, lca=False)))
        except:
            logging.critical('microbe: {}'.format(microbe))
            logging.critical('fmt: {}'.format(fmt))
            logging.critical('label: {}'.format(label))
            raise

    if LCA_FORMATTER in label:
        lineage = list(microbe.get_lineage(level='species'))
        while len(lineage) > 0:
            if str(lineage[-1]) != 'nan':
                label = label.replace(LCA_FORMATTER, str(lineage[-1]))
                break
            else:
                lineage = lineage[:-1]
        if len(lineage) == 0:
            logging.warning('All taxonomic levels are nans: {}'.format(microbe.get_lineage(level='species')))

    return label

@numba.jit(nopython=True, cache=True)
def fast_index(M, rows, cols):
    '''Fast index fancy indexing the matrix M. ~98% faster than regular
    fancy indexing
    M MUST BE C_CONTIGUOUS for this to actually help.
        If it is not C_CONTIGUOUS then ravel will have to copy the 
        array before it flattens it. --- super slow

    Parameters
    ----------
    M : np.ndarray 2-dim
        Matrix we are indexing at 2 dimensions
    rows, cols : np.ndarray
        rows and columns INDEX arrays. This will not work with bool arrays

    Returns
    -------
    np.ndarray
    '''
    return (M.ravel()[(cols + (rows * M.shape[1]).reshape(
        (-1,1))).ravel()]).reshape(rows.size, cols.size)

def toarray(x, dest=None, T=False):
    '''Converts `x` into a C_CONTIGUOUS numpy matrix if 
    the matrix is sparse. If it is not sparse then it just returns
    the matrix.

    Parameters
    ----------
    x : scipy.sparse, np.ndarray
        Array we are converting
    dest : np.ndarray
        If this is specified, send the array into this array. Assumes
        the shapes are compatible. Else create a new array
    T : bool
        If True, set the transpose

    Returns
    -------
    np.ndarray
    '''
    if scipy.sparse.issparse(x):
        if dest is None:
            if T:
                dest = np.zeros(shape=x.T.shape, dtype=x.dtype)
            else:
                dest = np.zeros(shape=x.shape, dtype=x.dtype)
        if T:
            x.T.toarray(out=dest)
        else:
            x.toarray(out=dest)
        return dest
    else:
        if T:
            return x.T
        else:
            return x

def subsample_timeseries(T, sizes, approx=True):
    '''Subsample the time-series `T` into size in `sizes`. Note that
    this algorithm does not guarentee any timepoints to stay.

    The baseline algorithm calculates:
    d(R) = \sum_{i \neq j} 1 / (R[i] - R[j])^2
    for every *COMBINATION* of T of size size. If |T| = 75 and size = 45,
    then there are 45! * (75 - 45)! = 3.2e88 combinations - this is not 
    doable.

    Approximation
    -------------
    To approximate this, we divide the current size of the of the elements 
    into `size+1` (approximately) equal intervals
    Example:
        T = [0, 0.5, 1, 2, 4, 4.5, 5, 7, 8, 10]
        len(T) = 10
        sizes = [8, 6]

        time_indexes for T:
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9

        Make 8 point interval:
            Need to delete 2 time points,
            10 - 8 = 2
            floor(10 / 3) = 3
            Delete every 3 elements
            0 , 1 , 3, 4, 6, 7, 9, 10

            T_8 = []

    Parameters
    ----------
    T : array_like
        These are the times that we want to subsample
    sizes : int, array(int)
        These are the sizes we want to subsample to. 

    Returns
    -------
    list(np.ndarray(sizes))
        A list of time series for each size in decreasing size order
    '''
    if not isarray(T):
        raise TypeError('`T` ({}) must be an array'.format(type(T)))
    if isint(sizes):
        sizes = [sizes]
    elif isarray(sizes):
        for size in sizes:
            if not isint(size):
                raise TypeError('Each size in `sizes` ({}) must be an int ({})'.format(
                    type(size), sizes))
    else:
        raise ValueError('`sizes` ({}) must be an int or an array f ints'.format(type(sizes)))

    T = np.asarray(T)
    sizes = np.unique(np.asarray(sizes, dtype=int))
    sizes[::-1].sort()
    ret = []
    l = len(T)

    prev_tp = np.arange(len(T))
    for n in sizes:
        spacings = []
        subsets = []

        for subset in itertools.combinations(prev_tp, n):
            subsets.append(subset)
            subset = list(subset)
            subset = [-1] + subset + [l+1]
            subset = np.array(subset)
            spacings.append((1/np.power(scipy.spatial.distance.pdist(
                subset[:, np.newaxis]), 2)).sum())

        idxs = np.array(subsets[spacings.index(min(spacings))])
        ret.append(T[idxs])
        prev_tp = idxs

    return ret
