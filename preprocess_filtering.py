'''This is the module that implements the filtering for the ASVs
'''
import numpy as np
import logging
import copy

import matplotlib.pyplot as plt
import seaborn as sns

import config
import pylab as pl

UNHEALTHY_SUBJECTS = ['2','3','4','5']
HEALTHY_SUBJECTS = ['6','7','8','9','10']

def at_least_counts(subjset, fparams):
    '''For each ASV in the subjectset `subjset`, delete all ASVs that
    do not have at least an average minimum number of counts `min_avg_counts`
    for less than `min_num_subjects` subjects.

    It will only look at the abundances after the colonization period, which is
    indicated by the parameter `start_day`, where `start_day` is the time to
    start looking at the data. For example, if our colinization duration is
    10 days, then we set `start_day=10`.

    Parameters
    ----------
    subjset : str, pylab.base.SubjectSet
        This is the SubjectSet object that we are doing the filtering on
        If it is a str, then it is the location of the saved object.
    fparams : config.FilteringConfig
        Parameters for the filtering

    Returns
    -------
    pylab.base.SubjectSet
        This is the filtered subject set.

    Raises
    ------
    ValueError
        If types are not valid or values are invalid
    '''
    # Type checking
    if pl.isstr(subjset):
        subjset = pl.SubjectSet.load(subjset)
    if not pl.issubjectset(subjset):
        raise TypeError('`subjset` ({}) must be a pylab.base.SubjectSet'.format(
            type(subjset)))
    if not pl.isnumeric(fparams.MIN_COUNTS):
        raise TypeError('`fparams.MIN_COUNTS` ({}) must be a numeric'.format(
            type(fparams.MIN_COUNTS)))
    if fparams.MIN_COUNTS < 0:
        raise ValueError('`fparams.MIN_COUNTS` ({}) must be >= 0')
    if not pl.isint(fparams.MIN_NUM_SUBJECTS):
        raise TypeError('`fparams.MIN_NUM_SUBJECTS` ({}) must be an int'.format(
            type(fparams.MIN_NUM_SUBJECTS)))
    if fparams.MIN_NUM_SUBJECTS > len(subjset) or fparams.MIN_NUM_SUBJECTS < 0:
        raise ValueError('`fparams.MIN_NUM_SUBJECTS` ({}) has an invalid value'.format(
            fparams.MIN_NUM_SUBJECTS))
    if fparams.COLONIZATION_TIME is None:
        fparams.COLONIZATION_TIME = 0.
    if not pl.isnumeric(fparams.COLONIZATION_TIME):
        raise TypeError('`fparams.COLONIZATION_TIME` ({}) must be a numeric'.format(
            type(fparams.COLONIZATION_TIME)))
    if fparams.COLONIZATION_TIME < 0:
        raise ValueError('`fparams.COLONIZATION_TIME` ({}) must be >= 0'.format(
            fparams.COLONIZATION_TIME))

    # Parameters are now valid, now do the filtering
    talley = np.zeros(len(subjset.asvs), dtype=int)
    for subj in subjset:
        matrix = subj.matrix(min_rel_abund=None)['raw']

        tidx = None
        for idx, t in enumerate(subj.times):
            if t >= fparams.COLONIZATION_TIME:
                tidx = idx
                break
        if tidx is None:
            raise ValueError('`colonization_time` ({}) was not found')
        M = matrix[:, tidx:]
        oidxs = []
        for oidx in range(M.shape[0]):
            if np.any(M[oidx, :] >= fparams.MIN_COUNTS):
                oidxs.append(oidx)
        talley[oidxs] += 1

    invalid_oidxs = np.where(talley < fparams.MIN_NUM_SUBJECTS)[0]
    invalid_oids = subjset.asvs.ids.order[invalid_oidxs]
    subjset.pop_asvs(invalid_oids)
    return subjset

def consistency(subjset, dtype, threshold, min_num_consecutive, colonization_time=None, 
    min_num_subjects=1, union_both_consortia=False):
    '''Filters the subjects by looking at the consistency of the 'dtype', which can
    be either 'raw' where we look for the minimum number of counts, 'rel', where we
    look for a minimum relative abundance, or 'abs' where we look for a minium 
    absolute abundance.

    There must be at least `threshold` for at least
    `min_num_consecutive` consecutive timepoints for at least
    `min_num_subjects` subjects for the ASV to be classified as valid.

    If a colonization time is specified, we only look after that timepoint

    Parameters
    ----------
    subjset : str, pylab.base.SubjectSet
        This is the SubjectSet object that we are doing the filtering on
        If it is a str, then it is the location of the saved object.
    dtype : str
        This is the string to say what type of data we are thresholding. Options
        are 'raw', 'rel', or 'abs'.
    threshold : numeric
        This is the threshold for either counts, relative abundance, or
        absolute abundance
    min_num_consecutive : int
        Number of consecutive timepoints to look for
    colonization_time : numeric
        This is the time we are looking after for colonization. If None we assume 
        there is no colonization time.
    min_num_subjects : int, str
        This is the minimum number of subjects this needs to be valid for.
        If str, we accept 'all', which we set that automatically.
    union_both_consortia : bool
        If True, run consistency with the passed in parameters for both healthy and 
        unhealthy consortium independently and then return the union of that operation.

    Returns
    -------
    pylab.base.SubjectSet
        This is the filtered subject set.

    Raises
    ------
    ValueError
        If types are not valid or values are invalid
    '''
    if not pl.isstr(dtype):
        raise TypeError('`dtype` ({}) must be a str'.format(type(dtype)))
    if dtype not in ['raw', 'rel', 'abs']:
        raise ValueError('`dtype` ({}) not recognized'.format(dtype))
    if not pl.issubjectset(subjset):
        raise TypeError('`subjset` ({}) must be a pylab.base.SubjectSet'.format(
            type(subjset)))
    if not pl.isnumeric(threshold):
        raise TypeError('`threshold` ({}) must be a numeric'.format(type(threshold)))
    if threshold <= 0:
        raise ValueError('`threshold` ({}) must be > 0'.format(threshold))
    if not pl.isint(min_num_consecutive):
        raise TypeError('`min_num_consecutive` ({}) must be an int'.format(
            type(min_num_consecutive)))
    if min_num_consecutive <= 0:
        raise ValueError('`min_num_consecutive` ({}) must be > 0'.format(min_num_consecutive))
    if colonization_time is None:
        colonization_time = 0
    if not pl.isnumeric(colonization_time):
        raise TypeError('`colonization_time` ({}) must be a numeric'.format(
            type(colonization_time)))
    if colonization_time < 0:
        raise ValueError('`colonization_time` ({}) must be >= 0'.format(colonization_time))
    if min_num_subjects is None:
        min_num_subjects = 1
    if min_num_subjects == 'all':
        min_num_subjects = len(subjset)
    if not pl.isint(min_num_subjects):
        raise TypeError('`min_num_subjects` ({}) must be an int'.format(
            type(min_num_subjects)))
    if min_num_subjects > len(subjset) or min_num_subjects <= 0:
        raise ValueError('`min_num_subjects` ({}) value not valid'.format(min_num_subjects))
    if not pl.isbool(union_both_consortia):
        raise TypeError('`union_both_consortia` ({}) must be a bool'.format(
            type(union_both_consortia)))
    
    if union_both_consortia:
        asvs_to_keep = set()
        for subjs_to_delete in [HEALTHY_SUBJECTS, UNHEALTHY_SUBJECTS]:
            subjset_temp = copy.deepcopy(subjset)
            subjset_temp.pop_subject(subjs_to_delete)
            subjset_temp = consistency(subjset_temp, dtype=dtype,
                threshold=threshold, min_num_consecutive=min_num_consecutive,
                colonization_time=colonization_time, min_num_subjects=min_num_subjects,
                union_both_consortia=False)
            for asv_name in subjset_temp.asvs.names:
                asvs_to_keep.add(asv_name)
        to_delete = []
        for aname in subjset.asvs.names:
            if aname not in asvs_to_keep:
                to_delete.append(aname)
    else:
        # Everything is fine, now we can do the filtering
        talley = np.zeros(len(subjset.asvs), dtype=int)
        for i, subj in enumerate(subjset):
            matrix = subj.matrix(min_rel_abund=None)[dtype]
            tidx_start = None
            for tidx, t in enumerate(subj.times):
                if t >= colonization_time:
                    tidx_start = tidx
                    break
            if tidx_start is None:
                raise ValueError('Something went wrong')
            matrix = matrix[:, tidx_start:]

            for oidx in range(matrix.shape[0]):
                consecutive = 0
                for tidx in range(matrix.shape[1]):
                    if matrix[oidx,tidx] >= threshold:
                        consecutive += 1
                    else:
                        consecutive = 0
                    if consecutive >= min_num_consecutive:
                        talley[oidx] += 1
                        break

        invalid_oidxs = np.where(talley < min_num_subjects)[0]
        to_delete = subjset.asvs.ids.order[invalid_oidxs]
    subjset.pop_asvs(to_delete)
    return subjset

def plot_asv(subjset, asv, fparams, legend, fig=None, title_format=None,
    suptitle_format=None, yscale_log=True, matrixes=None, read_depthses=None, 
    qpcrses=None, xlabel='Days', ylabel='CFUs/g', adjust_top=0.9, suptitle_fontsize=20):
    '''Plot the ASV `oidx` in the SubjectSet `subjset`. We make a new axis for
    each subject in the subject set. We use the filtering parameters specified 
    in `fparams` to get the limit of detection.

    Parameters
    ----------
    subjset : pylab.base.SubjectSet
        This is the object that contains all of the data
    oidx : int, str, pylab.base.ASV
        This is the identifier for the ASV we are plotting. You can use the 
        id, index, name, or pass in the actual object to get it.
    fparams : config.FilteringConfig, None
        These are the filtering parameters that were used during filtering in
        the function `consistency`. 
    legend : bool
        If True, we add a legend
    fig : matplotlib.pyplot.Figure, None
        This is the figure that we are plotting on. If this is None then we 
        create our own.
    title_format, suptitle_format : str, None
        This is the format for the title on each axis and for the figure, respectively. 
        We send this through `pylab.asvname_formatter`. We also add the formatting 
        keywords:
            - '%(sid)s': Subject ID
            - '%(sidx)s': Subject index
            - '%(sname)s': Subject name
    matrix : None, np.ndarray
        This is the matrix of the data in time order
    read_depths : np.ndarray, None
        These are the read_depths in time order
    adjust_top : float
        subplot_adjust keyword for `top`

    Returns
    -------
    matplotlib.pyplot.Figure
    '''
    # Check parameters
    if not pl.issubjectset(subjset):
        raise TypeError('`subjset` ({}) is not a pylab.SubjectSet'.format(type(subjset)))
    if fparams is not None:
        if type(fparams) is not config.FilteringConfig:
            raise TypeError('`fparams` ({}) is not a config.FilteringConfig'.format(
                type(fparams)))
    if not pl.isbool(legend):
        raise TypeError('`legend` ({}) mnust be a bool'.format(type(legend)))
    if fig is None:
        fig = plt.figure()
    if type(fig) is not plt.Figure:
        raise TypeError('`fig` ({}) must be a matplotlib.pyplot.Figure'.format(
            type(fig)))
    if not pl.isstr(title_format) and title_format is not None:
        raise TypeError('`title_format` ({}) must be a str'.format(type(title_format)))
    if not pl.isstr(suptitle_format) and suptitle_format is not None:
        raise TypeError('`suptitle_format` ({}) must be a str'.format(type(suptitle_format)))

    # Define the layout of the Axes
    if len(subjset) == 0:
        raise ValueError('There must be at least 1 subject in `subjset`')
    elif len(subjset) == 1:
        axis_layout = (1,1)
    elif len(subjset) == 2:
        axis_layout = (2,1)
    elif len(subjset) in [3,4]:
        axis_layout = (2,2)
    elif len(subjset) in [5,6]:
        axis_layout = (3,2)
    elif len(subjset) == 9:
        axis_layout = (3,3)
    else:
        raise ValueError('`Unknow layout for number of subject greater than 6 ({})'.format(
            len(subjset)))

    # Get the index of the ASV
    try:
        asv = subjset.asvs[asv]
    except:
        raise IndexError('`asv` ({}) was not found in the ASVSet'.format(asv))
    oidx = asv.idx

    if suptitle_format is not None:
        suptitle = pl.asvname_formatter(suptitle_format, asv=asv, asvs=subjset.asvs)
    else:
        suptitle = None

    for sidx, subj in enumerate(subjset):
        ax = fig.add_subplot(*axis_layout + (sidx+1,))
        if title_format is not None:
            title = pl.asvname_formatter(title_format.replace(
                '%(sid)s', str(subj.id)).replace(
                '%(sidx)s', str(sidx)).replace( 
                '%(sname)s', str(subj.name)), asv=asv, asvs=subjset.asvs)
        else:
            title = None

        if matrixes is None:
            matrix = subjset.matrix()['abs']
        else:
            matrix = matrixes[sidx]
        if qpcrses is None:
            qpcrs = np.sum(matrix, axis=0)
        else:
            qpcrs = qpcrses[sidx]
        if read_depthses is None:
            read_depths = subj.read_depth()
        else:
            read_depths = read_depthses[sidx]
        

        palette = sns.color_palette()
        traj = matrix[oidx, :]
        times = subj.times
        min_abund = (1/read_depths) * qpcrs

        if fparams is not None:
            if fparams.DTYPE == 'rel':
                threshold = fparams.THRESHOLD * qpcrs
            elif fparams.DTYPE == 'raw':
                threshold = (fparams.THRESHOLD/read_depths) * qpcrs
            else:
                threshold = np.ones(len(times), type=float) * fparams.THRESHOLD
        else:
            threshold = None

        # ax.plot(times, min_abund, label='Single Count', color=palette[0], 
        #     marker='.', alpha=0.5, zorder=10)
        # ax.plot(times, threshold, label='threshold', color=palette[1],
        #     marker='.', alpha=0.5, zorder=9)
        # idxs = traj >= threshold
        # ax.scatter(times[idxs], traj[idxs], c='black', marker='.', s=45, zorder=100)
        # idxs = traj < threshold
        # ax.scatter(times[idxs], traj[idxs], c='red', marker='.', s=45, zorder=100)
        ax.plot(times, traj, label='data', color='black', zorder=50,
            linestyle=':', marker='x')


        # if c_m is not None:
        #     ax.axhline(y=c_m, color=palette[3], alpha=0.5, label=r'$c_m$')

        if sidx == 1 or len(subjset) == 1:
            if legend:
                ax.legend(bbox_to_anchor=(1.01,1))
        if yscale_log:
            ax.set_yscale('log')
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if subjset.perturbations is not None:
            pl.visualization.shade_in_perturbations(ax=ax, 
                perturbations=subjset.perturbations)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize)
    fig.tight_layout()
    fig.subplots_adjust(top=adjust_top)

    return fig
        
        





        


    