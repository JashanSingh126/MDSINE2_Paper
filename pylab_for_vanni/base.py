import numpy as np
import pickle
import pandas as pd
import logging

from . import util

'''
TODO:
    Adjust the reads when you add a new microbe after you have created the 
    MicrobeSet object in the Study
'''


DEFAULT_TAXA_NAME = 'NA'
TAX_IDXS = {'kingdom': 0, 'phylum': 1, 'class': 2,  'order': 3, 'family': 4, 
    'genus': 5, 'species': 6, 'strain': 7}
_TAX_REV_IDXS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'strain']

class Perturbation(util.Saveable):
    '''Base perturbation class.

    Paramters
    ---------
    start : float, int
        This is the start of the perturbation (inclusive)
    end : float, int
        This is the end of the perturbation (exclusive)
    name : str
        This is the name of the perturbation
    subjects_affected : iterable
        These are the subjects are affected by the perturbation
    pset : PerturbationSet
        The perturbation set this perturbation belongs to.
    '''
    def __init__(self, start, end, name, subjects_affected, pset):
        if not util.isnumeric(start):
            raise TypeError('`start` ({}) must be a numeric'.format(type(start)))
        if not util.isnumeric(end):
            raise TypeError('`end` ({}) must be a numeric'.format(type(end)))
        if end < start:
            raise ValueError('`end` ({}) must be >= `start` ({})'.format(end, start))
        if not util.isstr(name):
            raise TypeError('`name` ({}) must be a str'.format(type(name)))
        self.start = start
        self.end = end
        self.name = name
        self.pset = pset
        self.subjects_affected = set(list(subjects_affected))

        study = self.pset.study
        for subjname in subjects_affected:
            subj = study[subjname]
            subj.perturbations[self.name] = self
        study.perturbations[self.name] = self
    
    def __str__(self):
        return 'Perturbation {}\n\tstart: {}\n\tend:{}' \
            '\n\tsubjects_affected:{}\n\tStudy:{}'.format(
            self.name, self.start, self.end, self.subjects_affected,
            self.pset.study)

    def add_subject_affected(self, subjname):
        if subjname not in self.pset.study:
            raise ValueError('`Subject` ({}) not in the study'.format(subjname))
        self.subjects_affected.add(subjname)
        subject = self.pset.study[subjname]
        subjname.perturbations[self.name] = self

    def remove_subject_affected(self, subjname):
        if subjname not in self.pset.study:
            raise ValueError('`Subject` ({}) not in the study'.format(subjname))
        self.subjects_affected.remove(subjname)
        subject = self.pset.study[subjname]
        subject.perturbations.pop(self.name, None)

    def isactive(self, time):
        '''Returns a `bool` if the perturbation is on at time `time`.

        Parameters
        ----------
        time : float, int
            Time to check
        '''
        return time > self.start and time <= self.end

    def timetuple(self):
        '''Returns the time tuple of the start and end

        Paramters
        ---------
        None

        Returns
        -------
        2-tuple
            (start,end) as floats
        '''
        return (self.start, self.end)

    def remove(self):
        '''Remove itself from the subjects its affected
        '''
        study = self.pset.study
        study.perturbations.pop(self.name, None)
        for subjname in self.subjects_affected:
            subj = study[subjname]
            subj.pop(self.name, None)

        return self


class PerturbationSet(util.Saveable):
    '''
    '''
    def __init__(self, study):
        self._perts = {}
        self.study = study

    def __iter__(self):
        for k in self._perts:
            yield self._perts[k]

    def __contains__(self, k):
        return k in self._perts

    def __getitem__(self, k):
        return self._perts[k]

    def __len__(self):
        return len(self._perts)

    def items(self):
        '''Iterate over key, value pairs
        '''
        for k,v in self._perts.items():
            yield k,v

    def isempty(self):
        '''Returns True if there are no perturbations in this container
        '''
        return len(self) == 0

    def add_perturbation(self, name, start, end, subjects_affected='all'):
        
        if subjects_affected == 'all':
            subjects_affected = self.study.get_subject_names()
        
        for subjname in subjects_affected:
            if subjname not in self.study:
                raise ValueError('`subjname` ({}) not recognized in the study ({})'.format(
                    subjname, self.study.get_subject_names()))

        pert = Perturbation(start=start, end=end, name=name, 
            subjects_affected=subjects_affected, pset=self)
        self._perts[pert.name] = pert

    def get_names(self):
        return list(self._perts.keys())
    
    def pop_perturbation(self, name, notexist_ok=True):
        '''Remove the perturbation `name`

        Parameters
        ----------
        name : str, Perturbation
            This is the identifier of the perturbation
        notexist_ok : bool
            If False, it will raise an error if the name is not included in the perturbation
            set.
        '''
        if isperturbation(name):
            name = name.name
        if name not in self:
            if not notexist_ok:
                raise ValueError('`{}` not found in the perturbation set ({})'.format(
                    name, self.get_names()))

        pert = self[name]
        pert.remove()
        self._perts.pop(name, None)

    def pop_subject(self, name):
        for pert in self:
            if name in pert.


class Mass(util.Saveable):
    
    def __init__(self, value):
        self.value = value
        self.scaling_factor = 1

    def set_scaling_factor(self, scaling_factor):
        raise NotImplementedError('Need to implement')


class qPCRMeasurement(Mass):
    '''Define a qPCR measurement.

    If dilution factor and mass are not specified, then we 
    assume that the CFUs are already normalized by those values

    Parameters
    ----------
    cfus : array
        These are the raw CFU values. If `mass` and `dilution_factor` are
        not specified, then we assume these are the cfus/g values. 
    mass : numeric
        This is the mass of the sample
    dilution_factor : numeric
        This is the dilution factor for the sample
        Example:
            If the sample was diluted to 1/100 of its original concentration,
            the dilution factor is 100, NOT 1/100.
    '''
    def __init__(self, cfus, mass=None, dilution_factor=None):
        self._raw_data = np.asarray(list(cfus))
        self.mass = mass
        self.dilution_factor = dilution_factor
        self.scaling_factor = 1
        self.recalculate_parameters()
        Mass.__init__(self, value=self._gmean)

    def recalculate_parameters(self):
        if len(self._raw_data) == 0:
            return

        data = self._raw_data
        if self.mass is not None:
            data = data/self.mass
        if self.dilution_factor is not None:
            data = data*self.dilution_factor
        self.data = data * self.scaling_factor
        self.log_data = np.log(self.data)
        
        self.loc = np.mean(self.log_data)
        self.scale = np.std(self.log_data)
        self._gmean = np.prod(self.data)**(1/len(self.data))
        self.value = self._gmean

    def add(self, raw_data):
        '''Add extra measurements `raw_data`

        Parameters
        ----------
        raw_data : float, array_like
            This is the measurement to add
        '''
        self._raw_data = np.append(self._raw_data, 
            np.asarray(list(raw_data)))
        self.recalculate_parameters()

    def set_scaling_factor(self, scaling_factor):
        '''Resets the scaling factor

        Parameters
        ----------
        scaling_factor : float, int
            This is the scaling factor to set everything to
        '''
        if scaling_factor <= 0:
            raise ValueError('The scaling factor must strictly be positive')
        self.scaling_factor = scaling_factor
        self.recalculate_parameters()

    def mean(self):
        '''Return the geometric mean
        '''
        return self._gmean


class Microbe(util.Saveable):
    '''Wrapper class for a single microbe

    Parameters
    ----------
    name : str
        Name given to the microbe 
    sequence : str
        Base Pair sequence
    idx : int
        The index that the asv occurs
    '''
    def __init__(self, name, idx, sequence=None):
        self.name = name
        self.sequence = sequence
        self.idx = idx
        if sequence is not None:
            self._sequence_as_array = np.array(list(sequence))
        else:
            self._sequence_as_array = None
        # Initialize the taxonomies to nothing
        self.taxonomy = {
            'kingdom': DEFAULT_TAXA_NAME,
            'phylum': DEFAULT_TAXA_NAME,
            'class': DEFAULT_TAXA_NAME,
            'order': DEFAULT_TAXA_NAME,
            'family': DEFAULT_TAXA_NAME,
            'genus': DEFAULT_TAXA_NAME,
            'species': DEFAULT_TAXA_NAME,
            'strain': DEFAULT_TAXA_NAME}
        self.id = id(self)

    def __getitem__(self,key):
        return self.taxonomy[key.lower()]

    def __eq__(self, val):
        '''Compares different microbes between each other. Checks all 
        of the attributes but the id

        Parameters
        ----------
        val : any
            This is what we are checking if they are equivalent
        '''
        if type(val) != Microbe:
            return False
        if self.name != val.name:
            return False
        if self.sequence != val.sequence:
            return False
        for k,v in self.taxonomy.items():
            if v != val.taxonomy[k]:
                return False
        return True

    def __str__(self):
        return 'Microbe\n\tid: {}\n\tidx: {}\n\tname: {}\n' \
            '\ttaxonomy:\n\t\tkingdom: {}\n\t\tphylum: {}\n' \
            '\t\tclass: {}\n\t\torder: {}\n\t\tfamily: {}\n' \
            '\t\tgenus: {}\n\t\tspecies: {}\n\t\tspecies: {}'.format(
            self.id, self.idx, self.name,
            self.taxonomy['kingdom'], self.taxonomy['phylum'],
            self.taxonomy['class'], self.taxonomy['order'],
            self.taxonomy['family'], self.taxonomy['genus'],
            self.taxonomy['species'], self.taxonomy['strain'])

    def set_taxonomy(self, tax_kingdom=None, tax_phylum=None, tax_class=None,
        tax_order=None, tax_family=None, tax_genus=None, tax_species=None,
        tax_strain=None):
        '''Sets the taxonomy of the parts that are specified

        Parameters
        ----------
        tax_kingdom, tax_phylum, tax_class, tax_order, tax_family, tax_genus, tax_strain : str
            'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'strain'
            Name of the taxa for each respective level
        '''
        if tax_kingdom is not None:
            self.taxonomy['kingdom'] = str(tax_kingdom)
        if tax_phylum is not None:
            self.taxonomy['phylum'] = str(tax_phylum)
        if tax_class is not None:
            self.taxonomy['class'] = str(tax_class)
        if tax_order is not None:
            self.taxonomy['order'] = str(tax_order)
        if tax_family is not None:
            self.taxonomy['family'] = str(tax_family)
        if tax_genus is not None:
            self.taxonomy['genus'] = str(tax_genus)
        if tax_species is not None:
            self.taxonomy['species'] = str(tax_species)
        if tax_strain is not None:
            self.taxonomy['strain'] = str(tax_strain)

        return self

    def get_lineage(self, level=None, lca=False):
        '''Returns a tuple of the lineage in order from Kingdom to the level
        indicated. Default value for level is to return everything including the name. 4
        If `lca` is True, then we return the lineage up to `level` where it is specified 
        (no nans)

        Parameters
        ----------
        level : str, Optional
            The taxonomic level you want the lineage until
            If nothing is provided, it returns the entire taxonomic lineage
            Example:
                level = 'class'
                returns a tuple of (kingdom, phylum, class)
        lca : bool
            Least common ancestor
        Returns
        -------
        str
        '''
        a =  (self.taxonomy['kingdom'], self.taxonomy['phylum'], self.taxonomy['class'],
            self.taxonomy['order'], self.taxonomy['family'], self.taxonomy['genus'],
            self.taxonomy['species'], self.taxonomy['strain'], self.name)

        if level is None:
            a = a
        if level == 'microbe':
            a = a
        elif level == 'strain':
            a = a[:-1]
        elif level == 'species':
            a = a[:-2]
        elif level == 'genus':
            a = a[:-3]
        elif level == 'family':
            a = a[:-4]
        elif level == 'order':
            a = a[:-5]
        elif level == 'class':
            a = a[:-6]
        elif level == 'phylum':
            a = a[:-7]
        elif level == 'kingdom':
            a = a[:-8]
        else:
            raise ValueError('level `{}` was not recognized'.format(level))

        if lca:
            i = len(a)-1
            while (type(a[i]) == float) or (a[i] == DEFAULT_TAXA_NAME):
                i -= 1
            a = a[:i+1]
        return a
    
    def get_taxonomy(self, level, lca=False):
        '''Get the taxonomy at the level specified

        Parameters
        ----------
        level : str
            This is the level to get
            Valid responses: 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'strain'
        lca : bool
            If True and the specified tax level is not specified, then supstitute it with
            the next highest taxonomy that's 
        '''
        return self.get_lineage(level=level, lca=lca)[-1]

    def tax_is_defined(self, level):
        '''Whether or not the microbe is defined at the specified taxonomic level

        Parameters
        ----------
        level : str
            This is the level to get
            Valid responses: 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'strain'
        
        Returns
        -------
        bool
        '''
        try:
            tax = self.taxonomy[level]
        except:
            raise KeyError('`tax` ({}) not defined. Available taxs: {}'.format(level, 
                list(self.taxonomy.keys())))
        return (type(tax) != float) and (tax != DEFAULT_TAXA_NAME)


class MicrobeSet(util.Saveable):
    '''Wraps a set of `Microbe` objects. You can get the Microbe object via the
    Microbe id, Microbe name, or Microbe sequence.
    Provides functionality for aggregating and getting subsets for lineages.

    Reading in a dataframe
    ----------------------
    The input dataframe is as follows:
        index is set to the name of the microbe
        Columns are 'sequence', 'kingdom', 'phylum', 'class', 'family'

    '''

    def __init__(self, df=None, use_sequences=True):
        '''Load data from a dataframe

        Assumes the frame has the following columns:
            - sequence
            - name
            - taxonomy
                * kingdom, phylum, class, order, family, genus, species, strain

        Parameters
        ----------
        df - pandas.DataFrame, Optional
            DataFrame containing the required information (Taxonomy, sequence).
            If nothing is passed in, it will be an empty set.
        use_sequences : bool
            If True, Each microbe must have an associated sequence. Else there are no sequences
        '''
        if not isbool(use_sequences):
            raise TypeError('`use_sequences` ({}) must be a bool'.format(
                type(use_sequences)))
        self.use_sequences = use_sequences

        self.ids = CustomOrderedDict()
        self.names = CustomOrderedDict()
        if self.use_sequences:
            self.seqs = CustomOrderedDict()
        else:
            self.seqs = None
        self.index = []
        self._len = 0

        if df is not None:
            self.set_from_dataframe(df=df)

    def __contains__(self,key):
        try:
            self[key]
            return True
        except:
            return False

    def __getitem__(self,key):
        '''Get a microbe by either its sequence, name, index, or id

        Parameters
        ----------
        key : str, int
            Key to reference the microbe
        '''
        if ismicrobe(key):
            return key
        if key in self.ids:
            return self.ids[key]
        elif isint(key):
            return self.index[key]
        elif key in self.names:
            return self.names[key]
        elif ismicrobe(key):
            return key
        elif self.use_sequences:
            if key in self.seqs:
                return self.seqs[key]
        else:
            raise IndexError('`{}` ({}) was not found as a name, sequence, index, or id'.format(
                key, type(key)))

    def __iter__(self):
        '''Returns each Microbe object in order
        '''
        for microbe in self.index:
            yield microbe

    def __len__(self):
        '''Return the number of microbes in the MicrobeSet
        '''
        return self._len

    def clear(self):
        '''Delete all of the microbes
        '''
        self.ids = CustomOrderedDict()
        self.names = CustomOrderedDict()
        if self.use_sequences:
            self.seqs = CustomOrderedDict()
        else:
            self.seqs = None
        self.index = []
        self._len = 0

    def set_from_dataframe(self, df, reset=False):
        '''Set from a DataFrame

        Index: name of the microbe
        columns: sequence, kingdom, class, order, family, 
            genus, species, strain
        
        If any of the columns of the taxonomies are not in there
        then we set it to the default name.

        Parameters
        ----------
        df : pandas.DataFrame
            This is the dataframe we are setting from. We assume that the index (row name)
            of the dataframe is the name of the microbe
        reset : bool
            If True, clear all of the internal microbes before
            we add these microbes
        '''
        if reset:
            self.clear()

        for tax in _TAX_REV_IDXS:
            if tax not in df.columns:
                df[tax] = DEFAULT_TAXA_NAME

        # Add all of the microbess from the dataframe if necessary
        df = df.rename(str.lower, axis='columns')
        for name in df.index:
            if self.use_sequences and 'sequence' in df:
                seq = df['sequence'][name]
            else:
                seq = None
            self.add_microbe(
                name=str(name),
                sequence=seq)
            self.names[str(name)].set_taxonomy(
                tax_kingdom=df.loc[name]['kingdom'],
                tax_phylum=df.loc[name]['phylum'],
                tax_class=df.loc[name]['class'],
                tax_order=df.loc[name]['order'],
                tax_family=df.loc[name]['family'],
                tax_genus=df.loc[name]['genus'],
                tax_species=df.loc[name]['species'],
                tax_strain=df.loc[name]['strain'])

    @property
    def n_microbes(self):
        '''Alias for __len__
        '''
        return self._len

    def add_microbe(self, name, sequence=None):
        '''Adds an microbe to the set

        Parameters
        ----------
        name : str
            This is the name of the ASV
        sequence : str
            This is the sequence of the ASV
        '''
        m = Microbe(name=name, sequence=sequence, idx=self._len)
        self.ids[m.id] = m
        if self.use_sequences:
            self.seqs[m.sequence] = m
        self.names[m.name] = m
        self.index.append(m)

        # update the order of the Microbes
        self.ids.update_order()
        if self.use_sequences:
            self.seqs.update_order()
        self.names.update_order()
        self._len += 1

        return self

    def del_microbe(self, microbe):
        '''Deletes the microbe from the set. If there is a phylogenetic
        tree

        Parameters
        ----------
        microbe : str, int, Microbe
            Can either be the name, sequence, or the ID of the microbe
        '''
        # Get the ID
        microbe = self[microbe]
        oidx = self.ids.index[microbe.id]

        # Delete the microbe from everything
        # microbe = self[microbe]
        self.ids.pop(microbe.id, None)
        if self.use_sequences:
            self.seqs.pop(microbe.sequence, None)
        self.names.pop(microbe.name, None)
        self.index.pop(oidx)

        # update the order of the microbes
        self.ids.update_order()
        if self.use_sequences:
            self.seqs.update_order()
        self.names.update_order()

        # Update the indices of the microbes
        # Since everything points to the same object we only need to do it once
        for idx,microbe in enumerate(self.ids.values()):
            microbe.idx = idx

        self._len -= 1

        return self

    def taxonomic_similarity(self, mic1, mic2):
        '''Calculate the taxonomic similarity between mic1 and mic2
        Iterates through most broad to least broad taxonomic level and
        returns the fraction that are the same.

        Example:
            mic1.taxonomy = (A,B,C,D)
            mic2.taxonomy = (A,B,E,F)
            similarity = 0.5

            mic1.taxonomy = (A,B,C,D)
            mic2.taxonomy = (A,B,C,F)
            similarity = 0.75

            mic1.taxonomy = (A,B,C,D)
            mic2.taxonomy = (A,B,C,D)
            similarity = 1.0

            mic1.taxonomy = (X,Y,Z,M)
            mic2.taxonomy = (A,B,E,F)
            similarity = 0.0

        Parameters
        ----------
        mic1, mic2 : str, int
            The name, id, or sequence for the microbe
        '''
        if mic1 == mic2:
            return 1
        mic1 = self[mic1].get_lineage()
        mic2 = self[mic2].get_lineage()
        i = 0
        for a in mic1:
            if a == mic2[i]:
                i += 1
            else:
                break
        return i/9 # including microbe)


class CustomOrderedDict(dict):
    '''Order is an initialized version of self.keys() -> much more efficient
    index maps the key to the index in order

    order (list)
        - same as a numpy version of the keys in order
    index (dict)
        - Maps the key to the index that it was inserted in
    '''

    def __init__(self, *args, **kwargs):
        '''Extension of the OrderedDict

        Paramters
        ---------
        args, kwargs : Arguments
            These are extra arguments to initialize the baseline OrderedDict
        '''
        dict.__init__(self, *args, **kwargs)
        self.order = None
        self.index = None

    def update_order(self):
        '''This will update the reverse dictionary
        '''
        self.order = np.array(list(self.keys()))
        self.index = {}
        for i, asv in enumerate(self.order):
            self.index[asv] = i


class Sample(util.Saveable):
    '''Single sample

    Parameters
    ----------
    study : Study
        Study this is associated with
    sampleid : str, int
        This is the identifier of the sample
        subject, this is the subject
    subjectid : str
        This is the name of the subject
    time : numeric
        This is the time that the measurement took place
    All other arguments are considered meta-data and are stored
    in the meta-data parameter
    '''
    def __init__(self, study, sampleid, subjectid, time, **meta_data):

        self.study = study
        self.sampleid = sampleid
        self.subjectid = subjectid
        self.time = time
        self.reads = None
        self.mass = None
        self.meta_data = meta_data

    def add_mass(self, mass):
        if not issubclass(mass, Mass):
            raise TypeError('the mass object must be a Mass object')
        self.mass = mass

    def add_reads(self, reads):
        '''
        '''
        subject = self.study[self.subjectid]
        if len(reads) != len(subject.microbes):
            raise ValueError('`reads` must have the same number of elements')
        reads = np.asarray(reads)

    def add_meta_data(self, clear=True, **kwargs):
        '''Add metadata
        '''
        if clear:
            self.meta_data = {}
        for k,v in kwargs.items():
            self.meta_data[k] = v

    def del_microbe(self, microbe_index):
        '''delete the mcirobe index
        '''
        self.reads = np.delete(self.reads, microbe_index)
        

class Subject(util.Saveable):
    '''Data for a single subject. This acts as a container for Sample objects

    name : str
        Name of the subject
    id : int
        Python unique integer
    study : Study
        Study that the subject belongs to
    microbes : MicrobeSet
        Set of microbes
    perturbations : dict (str -> Perturbation)
        Maps the name of the perturbation to the perturbation object.
        Assume that all perturbations in this dictionary affect this 
        subject.
    samples : dict (obj -> Sample)
        Maps the name of the sample to the Sample object

    Parameters
    ----------
    study : SubjectSet
        This is the Study class (we have a reverse pointer)
    name : str
        This is the name of the subject
    Extra arguments
        Extra arguemnts are set in self.meta_data
    '''
    def __init__(self, study, name, **meta_data):
        if not isstudy:
            raise TypeError('`study` ({}) must be a Study type'.format(
                type(study)))
        self.name = name
        self.id = id(self)
        self.study = study
        self.microbes = self.study.microbes
        self.meta_data = meta_data
        self.perturbations = {}
        self.samples = {}
        self._sampleids = {}
        self.times = np.asarray([])

    def __len__(self):
        return len(self.samples)

    def __contains__(self, k):
        return k in self.samples or k in self._sampleids

    def __iter__(self):
        for k in self.samples:
            yield self.samples[k]

    def __getitem__(self, k):
        if k in self.samples:
            return self.samples[k]
        elif k in self._sampleids:
            return self._sampleids[k]:
        else:
            raise KeyError('key ({}) not recognized as a sample ID or a time'.format(
                k))

    def add_sample(self, sample):
        '''Add the sample

        self.samples is indexed by time - make sure that we redo it
        such that the iterator goes through in order
        '''
        temp = {}
        tempid = {}
        sampletime = sample.time
        added = False

        # Readd them in order
        for sid, s in self.samples.items():
            if s.time <= sample.time:
                temp[s.time] = s
                tempid[s.sampleid] = s
            if s.time > sample.time:
                if added:
                    temp[s.time] = s
                    tempid[s.sampleid] = s
                else:
                    temp[sample.time] = sample
                    tempid[sample.sampleid] = sample
                    added = True
        
        self.samples = temp
        self._sampleids = tempid
        self.times = np.asarray(list(self.samples.keys()))
        
    def items(self):
        for k,v in self.samples.items():
            yield k,v

    def pop_time(self, t):
        if t not in self.samples:
            return
        self.samples.pop(t, None)
        self.times = np.asarray(list(self.samples.keys()))
        self.times = np.sort(self.times)

    def get_masses(self, t):
        '''Return the masses or a single mass at the specified timepoint

        Parameters
        ----------
        t : numeric
            Time to return
        '''
        if t is not None:
            return self.samples[t].mass.value
        return np.asarray([sample.mass.value for sample in self.samples])

    def set_meta_data(self, clear=True, **kwargs):
        '''Set the meta-data for the subject. If `clear` is True then
        clear the current values for the meta_data
        '''
        if clear:
            self.meta_data = {}
        for k,v in kwargs.items():
            self.meta_data[k] = v

    def matrix(self, dtype):
        '''Make a numpy matrix out of our data - returns the raw reads,
        the relative abundance, and the absolute abundance.

        If there is no mass data, then the absolute abundance is set to None.

        Parameters
        ----------
        dtype : str
            What type of matrix to return:
                'reads': return the reads
                'rel': return the relative abundance
                'abs': return the absolute value
        '''
        if dtype not in ['reads', 'rel', 'abs']:
            raise ValueError('`dtype` ({}) not recognized'.format(dtype))
        
        # Make reads
        shape = (len(self.microbes), len(self.times))
        M = np.zeros(shape=shape, dtype=int)
        for i,t in enumerate(self.samples):
            M[:,i] = self.samples[t].reads
        if dtype == 'reads':
            return M

        # Make relative abundances
        read_depths = np.sum(M, axis=0)
        M = M/read_depths
        if dtype == 'rel':
            return M

        # make absolute abundances
        for i,t in enumerate(self.samples):
            sample = self.samples[t]
            if sample.mass is not None:
                M[:, i] *= sample.mass.value
            else:
                raise ValueError('dtype `abs` specified but sample ({}) does not ' \
                    'have a mass measurement'.format(sample.mass))

        return M

    def df(self, **kwargs):
        '''Returns a dataframe of the data - same as matrix

        Parameters
        ----------
        These are the parameters for `matrix`
        '''
        M = self.matrix(**kwargs)
        index = self.microbes.names.order
        times = self.times
        return pd.DataFrame(data=M, index=index, columns=times)

    def read_depth(self, t=None):
        '''Get the read depth at time `t`. If nothing is given then return all
        of them

        Parameters
        ----------
        t : int, float, Optional
            Get the read depth at this time. If nothing is provided, all of the read depths for this 
            subject are returned
        '''
        arr = np.sum(self.matrix(dtype='reads'), axis=0)
        if t is None:
            if t not in self.times:
                raise ValueError('time `{}` not found in times ({})'.format(t, self.times))
            idx = np.searchsorted(self.times, t)
            return arr[idx]
        else:
            return arr

    def cluster_by_taxlevel(self, dtype, lca, taxlevel, index_formatter=None, smart_unspec=True):
        '''Clusters the microbes into the taxonomic level indicated in `taxlevel`.

        Smart Unspecified
        -----------------
        If True, returns the higher taxonomic classification while saying the desired taxonomic level
        is unspecified. Example: 'Order ABC, Family NA'. Note that this overrides the `index_formatter`.

        Parameters
        ----------
        subj : pylab.base.Subject
            This is the subject that we are getting the data from
        lca : bool
            If a microbe is unspecified at the taxonomic level and `lca` is True, then it will
            cluster at the higher taxonomic level
        taxlevel : str, None
            This is the taxa level to aggregate the data at. If it is 
            None then we do not do any collapsing (this is the same as 'microbe')
        dtype : str
            This is the type of data to cluster. Options are:
                'reads': These are the counts
                'rel': This is the relative abundances
                'abs': This is the absolute abundance (mass * rel)
        index_formatter : str
            How to make the index using `util.microbename_formatter`. Note that you cannot
            specify anything at a lower taxonomic level than what youre clustering at. For 
            example, you cannot cluster at the 'class' level and then specify '%(genus)s' 
            in the index formatter.
            If nothing is specified then only return the specified taxonomic level
        '''
        # Type checking
        if not util.isstr(dtype):
            raise TypeError('`dtype` ({}) must be a str'.format(type(dtype)))
        if dtype not in ['reads', 'rel', 'abs']:
            raise ValueError('`dtype` ({}) not recognized'.format(dtype))
        if not util.isstr(taxlevel):
            raise TypeError('`taxlevel` ({}) must be a str'.format(type(taxlevel)))
        if taxlevel not in ['kingdom', 'phylum', 'class',  'order', 'family', 
            'genus', 'species', 'strain', 'microbe']:
            raise ValueError('`taxlevel` ({}) not recognized'.format(taxlevel))
        if index_formatter is None:
            index_formatter = taxlevel
        if index_formatter is not None:
            if not util.isstr(index_formatter):
                raise TypeError('`index_formatter` ({}) must be a str'.format(type(index_formatter)))
            
            for tx in TAX_IDXS:
                if tx in index_formatter and TAX_IDXS[tx] > TAX_IDXS[taxlevel]:
                    raise ValueError('You are clustering at the {} level but are specifying' \
                        ' {} in the `index_formatter`. This does not make sense. Either cluster' \
                        'at a lower tax level or specify the `index_formatter` to a higher tax ' \
                        'level'.format(taxlevel, tx))

        index_formatter = index_formatter.replace('%(microbe)s', '%(name)s')

        # Everything is valid, get the data dataframe and the return dataframe
        df = self.df(min_rel_abund=None)[dtype]
        cols = list(df.columns)
        cols.append(taxlevel)
        dfnew = pd.DataFrame(columns = cols).set_index(taxlevel)

        # Get the level in the taxonomy, create a new entry if it is not there already
        taxas = {} # lineage -> label
        for i, microbe in enumerate(self.microbes):
            row = df.index[i]
            tax = microbe.get_lineage(level=taxlevel, lca=lca)
            tax = tuple(tax)
            tax = str(tax).replace("'", '')
            if tax in taxas:
                dfnew.loc[taxas[tax]] += df.loc[row]
            else:
                if not microbe.tax_is_defined(taxlevel) and smart_unspec:
                    # Get the least common ancestor above the taxlevel
                    taxlevelidx = TAX_IDXS[taxlevel]
                    ttt = None
                    while taxlevelidx > -1:
                        if microbe.tax_is_defined(_TAX_REV_IDXS[taxlevelidx]):
                            ttt = _TAX_REV_IDXS[taxlevelidx]
                            break
                        taxlevelidx -= 1
                    if ttt is None:
                        raise ValueError('Could not find a single taxlevel: {}'.format(str(microbe)))
                    taxas[tax] = '{} {}, {} NA'.format(ttt.capitalize(), 
                        microbe.taxonomy[ttt], taxlevel.capitalize())
                else:
                    taxas[tax] = util.microbename_formatter(format=index_formatter, microbe=microbe, microbes=self.microbes, lca=lca)
                toadd = pd.DataFrame(np.array(list(df.loc[row])).reshape(1,-1),
                    index=[taxas[tax]], columns=dfnew.columns)
                dfnew = dfnew.append(toadd)
        
        return dfnew


class Study(util.Saveable):
    '''Holds data for all the subjects

    Paramters
    ---------
    microbes : MicrobeSet, Optional
        If you already have an MicrobeSet, you can just use that
    '''
    def __init__(self, microbes):
        self.id = id(self)
        self._subjects = {}
        self.perturbations = None
        self.mass_normalization_factor = None
        if not ismicrobeset(microbes):
            raise ValueError('If `microbes` ({}) is specified, it must be an MicrobeSet' \
                ' type'.format(type(microbes)))
        self.microbes = microbes

        self._meta_data = {}
        self._samples = {}

    def __getitem__(self, key):
        return self._subjects[key]

    def __len__(self):
        return len(self._subjects)

    def __iter__(self):
        for v in self._subjects.values():
            yield v

    def __contains__(self, key):
        return key in self._subjects

    def iloc(self, idx):
        '''Get the subject as an index

        Parameters
        ----------
        idx : int
            Index of the subject

        Returns
        -------
        pl.base.Subject
        '''
        for i,sid in enumerate(self._subjects):
            if i == idx:
                return self._subjects[sid]
        raise IndexError('Index ({}) not found'.format(idx))

    def add_subject(self, name):
        '''Create a subject with the name `name`

        Parameters
        ----------
        name : str
            This is the name of the new subject
        '''
        if name not in self._subjects:
            self._subjects[name] = Subject(name=name, study=self)
        return self

    def subj_names(self):
        return list(self._subjects.keys())

    def get_sample(self, sampleid):
        '''Get the sample based on the sampleid
        '''
        try:
            return self._samples[sampleid]
        except:
            raise KeyError('sampleid ({}) not found'.format(sampleid))

    def parse_sample_metadata_from_dataframe(self, df):
        '''Parse the metadata for each sample in the study.

        Once this is called, it creates the subject objects

        Outline of the dataframe
        ------------------------
        Index : SampleID
        Columns:
            required Columns:
                'subjectid' : This is the name of the subject that this sample belongs to
                'time' : This is the time that the sample was taken
            Other columns may be set on the dataframe. These are then put as metadata for the
            sample

        Parameters
        ----------
        df : pandas.DataFrame
            Object containing the data
        '''
        if type(df) != pd.DataFrame:
            raise ValueError('`df` ({}) must be a dataframe'.format(type(df)))
        for required_col in ['subjectid', 'time']
            if 'subjectid' not in df.columns:
                raise ValueError('`{}` must be in the dataframe columns: {}'.format(
                    required_col, df.columns))
        for sampleid in df.index:
            d = {}
            for col in df.columns:
                d[col] = df.loc[sampleid][col]

            self._samples[sampleid] = Sample(study=self, **d)

        # Create the subject objects
        subjnames = set([])
        for sampleid in self._samples:
            subjnames.add(self._samples[sampleid].subjectid)
        
        for subjname in subjnames:
            self.add_subject(name=subjname)
            if subjname in self._meta_data:
                self._subjects[subjname].set_meta_data(
                    **self._meta_data[subjname])
        
        for sampleid, sample in self._samples.items():
            subjname = sample.subjectid
            self._subjects[subjname].add_sample(sample=sample)

        # Add metadata for the subjects if necessary
        if len(self._meta_data) > 0:
            for subjname in self._meta_data:
                if subjname not in self._subjects:
                    raise ValueError('Subject name from metadata ({}) not found ' \
                        'in subjects ({})'.format(subjname, list(self._subjects.keys())))
                subj = self[subjname]
                subj.set_meta_data(**self._meta_data[subjname])

    def parse_subject_metadata_from_dataframe(self, df):
        '''Parse the metadata for each subject in the study (optional)

        Outline of the dataframe
        ------------------------
        Index : SubjectID
        Columns: Each column is optional here - only call this add if you want
            to add extra information for each subject. All of this information 
            goes into the meta_data parameter for the subject

        Parameters
        ----------
        df : pandas.DataFrame
            Object containing the data
        '''
        if type(df) != pd.DataFrame:
            raise ValueError('`df` ({}) must be a dataframe'.format(type(df)))
        for subjectid in df.index:
            d = {}
            for col in df.columns:
                d[col] = df.loc[sampleid][col]
            self._meta_data[subjectid] = d

    def parse_reads_from_dataframe(self, df):
        '''Parse the read data for each subject

        Outline of the dataframe
        ------------------------
        Index : Microbe ID
        Columns: Sample ID

        Parameters
        ----------
        df : pandas.DataFrame
            Object containing the data
        '''
        if type(df) != pd.DataFrame:
            raise ValueError('`df` ({}) must be a dataframe'.format(type(df)))
        
        names = self.microbes.names.order
        try:
            df = df.loc[names]
        except:
            raise ValueError('Failed when reindexing based on the order of the microbes ' \
                'in the MicrobeSet. This likely happened because there was a microbe in the ' \
                'MicrobeSet that was not in the passed in dataframe.')
        M = df.values
        for col, sampleid in enumerate(df.columns):
            if sampleid not in self._samples:
                raise ValueError('sampleid ({}) not recognized. You must parse the ' \
                    'metadata for each sample with the function ' \ 
                    '`Study.parse_sample_metadata_from_dataframe` before you call this function')
            sample = self._samples[sampleid]
            sample.add_reads(reads=M[:,col])

    def parse_masses_from_dataframe(self, df, mass_type='qpcr'):
        '''Parse the mass data for each subject

        qPCR mass type
        --------------
        Index : Sample ID
        Columns: anything - we assume these are the cfus
            optional columns:
                'mass' : this is the mass of the sample
                'dilution_factor' : this is the dilution factor of the sample

        Parameters
        ----------
        df : pandas.DataFrame
            Object containing the data
        '''
        if type(df) != pd.DataFrame:
            raise ValueError('`df` ({}) must be a dataframe'.format(type(df)))

        if mass_type == 'qpcr':
            dfmeta = None
            if 'mass' in df.columns:
                dfmeta = df['mass']
            if 'dilution_factor' in df.columns:
                if dfmeta is None:
                    dfmeta = df
                else:
                    dfmeta['dilution_factor'] = df['dilution_factor']
            df = df.drop(dfmeta.columns, axis=1)
            M = df.values
            for row, sampleid in enumerate(df.index):
                if sampleid not in self._samples:
                    raise ValueError('sampleid ({}) not recognized. You must parse the ' \
                        'metadata for each sample with the function ' \ 
                        '`Study.parse_sample_metadata_from_dataframe` before you call this function')
                sample = self._samples[sampleid]
                
                cfus = M[row, :]
                d = {}
                for col in dfmeta:
                    d[col] = dfmeta.loc[sampleid][col]
                mass = qPCRMeasurement(cfus=cfus, **d)
                sample.add_mass(mass)

        else:
            raise ValueError('`mass_type` ({}) not recognized'.format(mass_type))

    def pop_subjects(self, subjects):
        '''Remove the indicated subject id

        Parameters
        ----------
        subjs : list(str), str, int
            This is the subject name/s or the index/es to pop out.
            Return a new SubjectSet with the specified subjects removed.
        '''
        if not util.isarray(subjs):
            subjs = [subjs]

        names = self.names()
        d = []
        for subject in subjects:
            if util.isint(subject):
                subject = names[subject]
            self._subjects.pop(subject, None)
            d.append(subject)
        
        for subjname in d:
            for pert in slf.perturbations:
                if subjname in pert.subjects_affected:
                    pert.remove_subject_affected(subjname=subjname)

    def pop_microbes(self, mids):
        '''Delete the microbes indicated in oidxs. Updates the reads table and
        the internal MicrobeSet

        Parameters
        ----------
        mids : str, int, list(str/int)
            These are the identifiers for each of the microbe/s to delete
        '''
        if not isarray(mids):
            mids = [mids]

        for mid in mids:
            microbe = self.microbes[mid]
            midx = microbe.idx
            self.microbes.del_microbe(microbe=microbe.name)

            for subj in self:
                for sample in subj:
                    sample.del_microbe(microbe_index=midx)

        return self

    def pop_times(self, times, subjects='all'):
        '''Discard the times in `times` for the subjects listed in `subjects`.
        If a timepoint is not found in a subject, no error is thrown.

        Parameters
        ----------
        times : numeric, list(numeric)
            Time/s to delete
        subjects : str, int, list(int)
            The Subject ID or a list of subject IDs that you want to delete the timepoints
            from. If it is a str:
                'all' - delete from all subjects
        '''
        if not util.isarray(times):
            times = [times]
        if subjects == 'all':
            subjects = self.subj_names()
        if not util.isarray(subjects):
            subjects = [subjects]

        for time in times:
            for subjname in subjects:
                if subjname not in self:
                    raise ValueError('`subjname` ({}) not found in study'.format(
                        subjname))
                subject = self[subjname]
                subject.pop_time(t=time)

    def normalize_mass(self, max_value):
        '''Normalize the mass values such that the largest value is the max value
        over all the subjects

        Parameters
        ----------
        max_value : float, int
            This is the maximum mass value to
        '''
        if type(max_value) not in [int, float]:
            raise ValueError('max_value ({}) must either be an int or a float'.format(
                type(max_value)))

        if self.mass_normalization_factor is not None:
            logging.warning('mass is already rescaled. unscaling and rescaling')
            self.denormalize_mass()

        masses = []
        for subj in self:
            masses = np.append(masses, subj.get_masses())
        temp_max = np.max(masses)

        self.mass_normalization_factor = max_value/temp_max
        logging.info('max_value found: {}, scaling_factor: {}'.format(
            temp_max, self.mass_normalization_factor))

        for subj in self:
            for key in subj.sample:
                subj.sample[key].mass.set_scaling_factor(scaling_factor=
                    self.mass_normalization_factor)
        return self

    def denormalize_mass(self):
        '''Denormalizes the mass values if necessary
        '''
        if self.mass_normalization_factor is None:
            logging.warning('mass is not normalized. Doing nothing')
            return
        for subj in self:
            for t in subj.samples:
                subj.samples[t].mass.set_scaling_factor(scaling_factor=1)
        self.mass_normalization_factor = None
        return self

    def _matrix(self, dtype, agg, times):
        if dtype not in ['raw', 'rel', 'abs']:
            raise ValueError('`dtype` ({}) not recognized'.format(dtype))
        
        if agg == 'mean':
            aggfunc = np.nanmean
        elif agg == 'median':
            aggfunc = np.nanmedian
        elif agg == 'sum':
            aggfunc = np.nansum
        elif agg == 'max':
            aggfunc = np.nanmax
        elif agg == 'min':
            aggfunc = np.nanmin
        else:
            raise ValueError('`agg` ({}) not recognized'.format(agg))

        if util.isstr(times):
            all_times = []
            for subj in self:
                all_times = np.append(all_times, subj.times)
            all_times = np.sort(np.unique(all_times))
            if times == 'union':
                times = all_times

            elif times == 'intersection':
                times = []
                for t in all_times:
                    addin = True
                    for subj in self:
                        if t not in subj.times:
                            addin = False
                            break
                    if addin:
                        times = np.append(times, t)

            else:
                raise ValueError('`times` ({}) not recognized'.format(times))
        elif isarray(times):
            times = np.array(times)
        else:
            raise TypeError('`times` type ({}) not recognized'.format(type(times)))

        M = np.zeros(shape=(len(self.microbes), len(times)), dtype=float)
        for tidx, t in enumerate(times):
            temp = None
            for subj in self:
                if t not in subj.times:
                    continue
                if dtype == 'counts':
                    a = subj.samples[t].reads
                elif dtype == 'rel':
                    a = subj.samples[t].reads/np.sum(subj.samples[t].reads)
                else:
                    rel = subj.samples[t].reads/np.sum(subj.samples[t].reads)
                    if subj.samples[t].mass is None:
                        raise ValueError('`mass` for time `{}` in subject `{}` not defined'.format(
                            t, subj.name))
                    a = rel * subj.samples[t].mass.value
                if temp is None:
                    temp = (a.reshape(-1,1), )
                else:
                    temp = temp + (a.reshape(-1,1), )
            if temp is None:
                temp = np.zeros(len(self.asvs)) * np.nan
            else:
                temp = np.hstack(temp)
                temp = aggfunc(temp, axis=1)
            M[:, tidx] = temp

        return M, times

    def matrix(self, dtype, agg, times):
        '''Make a matrix of the aggregation of all the subjects in the subjectset

        Aggregation of subjects
        -----------------------
        What are the values for the ASVs? Set the aggregation type using the parameter `agg`. 
        These are the types of aggregations:
            'mean': Mean abundance of the ASV at a timepoint over all the subjects
            'median': Median abundance of the ASV at a timepoint over all the subjects
            'sum': Sum of all the abundances of the ASV at a timepoint over all the subjects
            'max': Maximum abundance of the ASV at a timepoint over all the subjects
            'min': Minimum abundance of the ASV at a timepoint over all the subjects

        Aggregation of times
        --------------------
        Which times to include? Set the times to include with the parameter `times`.
        These are the types of time aggregations:
            'union': Take  theunion of the times of the subjects
            'intersection': Take the intersection of the times of the subjects
        You can manually specify the times to include with a list of times. If times are not
        included in any of the subjects then we set them to NAN.

        Parameters
        ----------
        dtype : str
            What kind of data to return. Options:
                'counts': Count data
                'rel': Relative abundance
                'abs': Abundance data
        agg : str
            Type of aggregation of the values. Options specified above.
        times : str, array
            The times to include
        
        Returns
        -------
        np.ndarray(n_asvs, n_times)
        '''
        M, _ =  self._matrix(dtype=dtype, agg=agg, times=times)
        return M

    def df(self, *args, **kwargs):
        '''Returns a dataframe of the data in matrix. Rows are ASVs, columns are times
        '''
        M, times = self._matrix(*args, **kwargs)
        index = self.asvs.names.order
        return pd.DataFrame(data=M, index=index, columns=times)


def issubject(x):
    '''Checks whether the input is a subclass of Subject

    Parameters
    ----------
    x : any
        Input instance to check the type of Subject
    
    Returns
    -------
    bool
        True if `x` is of type Subject, else False
    '''
    return x is not None and issubclass(x.__class__, Subject)

def isstudy(x):
    '''Checks whether the input is a subclass of Study

    Parameters
    ----------
    x : any
        Input instance to check the type of Study
    
    Returns
    -------
    bool
        True if `x` is of type Study, else False
    '''
    return x is not None and issubclass(x.__class__, Study)

def ismicrobeset(x):
    '''Checks whether the input is a subclass of MicrobeSet

    Parameters
    ----------
    x : any
        Input instance to check the type of MicrobeSet
    
    Returns
    -------
    bool
        True if `x` is of type MicrobeSet, else False
    '''
    return x is not None and issubclass(x.__class__, MicrobeSet)

def ismass(x):
    '''Checks whether the input is a subclass of Mass

    Parameters
    ----------
    x : any
        Input instance to check the type of Mass
    
    Returns
    -------
    bool
        True if `x` is of type Mass, else False
    '''
    return x is not None and issubclass(x.__class__, Mass)

def isperturbation(x):
    '''Checks whether the input is a subclass of Perturbation

    Parameters
    ----------
    x : any
        Input instance to check the type of Perturbation
    
    Returns
    -------
    bool
        True if `x` is of type Perturbation, else False
    '''
    return x is not None and issubclass(x.__class__, Perturbation)