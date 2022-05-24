#generate semi-synthetic data trajectories
import mdsine2 as md2
import numpy as np
from mdsine2.names import STRNAMES
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def parse_arguments():

    parser = argparse.ArgumentParser(description="Arguments to generate"\
        " semi-synthetic data and perform inference")
    parser.add_argument("-s", "--seed", type=int, required=True)
    parser.add_argument("-f1", "--mcmc_file_loc", required=True,
        help='location of mcmc inference pkl file')
    parser.add_argument("-p", "--save_path", required=True,
        default="output/semisynthetic_experiments",
        help="location where the outputs are saved")

    #Optional parameters
    parser.add_argument("-a0_l","--negbin_a0_low", required=False, default=1e-6,
    	 type=float, help="parameter a0 of neg binomial distribution for low noise")
    parser.add_argument("-a1_l", "--negbin_a1_low", required=False, default=3e-6,
    	 type=float, help="parameter a1 of neg binomial distribution for low noise")
    parser.add_argument("-a0_m","--negbin_a0_med", required=False, default=1e-5,
    	 type=float, help="parameter a0 of neg binomial distribution for medium noise")
    parser.add_argument("-a1_m", "--negbin_a1_med", required=False, default=1.5e-2,
    	 type=float, help="parameter a1 of neg binomial distribution for medium noise")
    parser.add_argument("-a0_h","--negbin_a0_high", required=False, default=1e-4,
    	 type=float, help="parameter a0 of neg binomial distribution for high noise")
    parser.add_argument("-a1_h", "--negbin_a1_high", required=False, default=9e-2,
    	 type=float, help="parameter a1 of neg binomial distribution for high noise")

    parser.add_argument("-ns_l", "--noise_scale_low", required=False, default=0.01,
    	type=float, help="scale for low noise level")
    parser.add_argument("-ns_m", "--noise_scale_medium", required=False, default=0.15,
    	type=float, help="scale for medium noise level")
    parser.add_argument("-ns_h", "--noise_scale_high", required=False, default=0.3,
    	type=float, help="scale for high noise level")
    parser.add_argument("-rd", "--read_depth", required=False, default=75000,
    	type=int, help="read depth")

    parser.add_argument("-v", "--process_var", required=False, default=0.01,
    	type=float, help="the process variance for the gLV model")
    parser.add_argument("-t", "--dt", required=False, default=0.01,
    	type=float, help="delta t for forward simulating trajectory")

    return parser.parse_args()

def plot_trajectories(dset, seed, dtype, path, noise=None):
    """generates axes to plot the trajectories of OTUs

       (pl.Base.Study or pl.synthetic.Synthetic) dset : class containing
           information about the OTUs
       (int) seed
       (str) dtype : specifies where dset is Study or Synthetic
       (str) noise : noise level; can be low, medium, high
       (str) path : location of the folder where the results are saved

       @returns
       ------------
       (dict): (str) OTU ID -> (plt, figure, plt.axes)
    """

    taxa = dset.taxa
    axes_dict = {}
    for taxon in taxa:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("Days")
        ax.set_ylabel("CFUs/g")
        if noise is None:
            ax.set_title(md2.pylab.base.taxaname_for_paper(taxon, taxa) +
               ", seed {}".format(seed))
        else:
            ax.set_title(md2.pylab.base.taxaname_for_paper(taxon, taxa)+
                ", {} noise, seed {}".format(noise, seed))
        axes_dict[taxon.name] = (ax, fig)

    subjects = None
    if dtype =="study":
        subjects = dset
    elif dtype == "synthetic":
        subjects = dset._data
    x = dset.times
    for sub in subjects:
        d = subjects[sub]
        label = "mouse {}".format(sub)
        if dtype == "study":
            d = sub.matrix()["abs"]
            label = "mouse {}".format(sub.name)
            x = sub.times
        for i in range(d.shape[0]):
            key = taxa[i].name
            ax, fig = axes_dict[key]
            ax.plot(x, d[i, :], label=label)

    for key in axes_dict:
        ax, fig = axes_dict[key]
        #ax.set_ylim((1e-15, 1e12))
        ax.set_yscale("log")
        ax.legend()
        fig.savefig(path / "{}.pdf".format(key))
        plt.close()

def generate_semi_synthetic_trajectory(mcmc, seed, name, pvar, dt, n_subjects=None,
    plot=True, path = None, set_times=True, min_bf=1):
    """ generates (semi-synthetic)latent state trajectory using results from
        real data

        @parameters
        -----------------
        (pylab.inference.BaseMCMC) mcmc: results of the mcmc inference
        (int) seed
        (str) name : name assigned to the graph
        (float) dt : Delta t used in integrating the gLV model
        (float) pvar : process variance for the gLV
        (int) n_subjects : number of subjects in the semi-synthetic data; if
                           None, then it is equal to the number of subjects in
                           real data
        (bool) plot : If True, make plots of the trajectory and save it
        (str) path : path to the folder where the trajectory plots are
                         saved
        (bool) set_times : If True, then the sampled time-points in the union
                           of time-points for individual subjects
        (float) min_bf: minimum bayes factor needed for perturbation/interaction
                        to be used in generation of synthetic data

        @returns
        ------------------
        (md2.pylab.synthetic.Synthetic)

    """

    syn = md2.synthetic.make_semisynthetic(mcmc, seed, min_bayes_factor=min_bf,
        name=name, set_times=set_times)
    if n_subjects is not None:
        syn.set_subjects(["{}".format(i+1) for i in range(n_subjects)])

    #generate the trajectories
    syn.generate_trajectories(dt=dt, init_dist=md2.variables.Uniform(low=1e5,
       high=1e7), processvar=md2.model.MultiplicativeGlobal(pvar**2))
    if plot:
        savepath = path / "seed_{}".format(seed) / "trajectory_no_noise"
        savepath.mkdir(parents=True, exist_ok=True)
        plot_trajectories(syn, seed, "synthetic", savepath)

    return syn

def compute_variance(noisy_traj, latent_traj, seed, noise_level, path):
    """
       Computes the empirical variance of the noisy trajectory and
       the mean square different between the noisy trajectory and latent trajectory

       @parameters
       -------------------
       (pl.Base.Study) noisy_traj
       (pl.synthetic.Synthetic) latent_traj

       @returns
       (np.array) variance for each OTU, (np.array) mean square difference for
       each OTU
    """

    var_dict = {}
    diff_dict = {}
    all_taxa = noisy_traj.taxa

    for subj in noisy_traj:
        noisy_traj_subj = subj.matrix()["abs"]
        latent_traj_subj = latent_traj._data[subj.name]

        variance = np.std(noisy_traj_subj, axis=1)
        difference = np.sqrt(np.mean(np.square(noisy_traj_subj - latent_traj_subj),
            axis=1))

        var_dict["mouse {}".format(subj.name)] = variance
        diff_dict["mouse {}".format(subj.name)] = difference

    row_names = [taxon.name for taxon in all_taxa]
    df_var = pd.DataFrame.from_dict(var_dict)
    df_diff = pd.DataFrame.from_dict(diff_dict)

    df_var.index = row_names
    df_diff.index = row_names

    savepath = path / "seed_{}".format(seed) / "variance"
    savepath.mkdir(parents=True, exist_ok=True)
    df_var.to_csv(savepath / "{}_variance.csv".format(noise_level), sep=",")
    df_diff.to_csv(savepath / "{}_difference.csv".format(noise_level), sep=",")


def add_measurement_noise(syn, seed, noise_params_dict, noise_level, syn_name,
    path):
    """adds measurement noise (qPCR + read count) to synthetic trajectories

       (pl.synthetic.Synthetic) syn: object associated with synthetic data
       (int) seed
       (dict) noise_params_dict : dictionary containing the parameters for
            distributions associated with qPCR and read count add_measurements
       (str) noise_level
       (str) syn_name : name of the graph
       (Path) path : location where the outputs are saved

        @returns
        (pl.base.Study)
    """

    print("Adding measurement noise for seed {} and noise level : {}".format(
        seed, noise_level))
    syn_study = syn.simulateMeasurementNoise(a0=noise_params_dict["a0"],
       a1=noise_params_dict["a1"], qpcr_noise_scale=noise_params_dict["scale"],
       approx_read_depth=noise_params_dict["depth"], name=syn_name)

    metadata_path = path / "seed_{}".format(seed) / "metadata" / "{}_noise".format(
       noise_level)
    traj_path = path / "seed_{}".format(seed) / "trajectory_{}_noise".format(
       noise_level)
    metadata_path.mkdir(parents=True, exist_ok=True)
    traj_path.mkdir(parents=True, exist_ok=True)

    export_metadata(syn_study, metadata_path)
    plot_trajectories(syn_study, seed, "study", traj_path, noise=noise_level)
    compute_variance(syn_study, syn, seed, noise_level, path)

    return syn_study

def export_metadata(study, path):
    """exports the metadata associated with dset

       (pl.base.Study) study
       (Path) pathname : location of the folder where the metadata is saved
    """

    study.write_metadata_to_csv(path=path / 'metadata.tsv')
    df = study.write_qpcr_to_csv(path=path / 'qpcr.tsv')

    if study.perturbations is not None:
        study.write_perturbations_to_csv(path=path / 'perturbations.tsv')
    study.write_reads_to_csv(path=path / 'reads.tsv')
    study.taxa.write_taxonomy_to_csv(path=path / 'taxonomy.tsv')

def combined_plot(latent_data, low_noise_data, med_noise_data, high_noise_data,
    seed, path):

    """plots the trajectories for latent state, low noise, medium noise,
       and high noise together in a single plot
    """

    savepath = path / "seed_{}".format(seed) / "combined_plot"
    all_taxa = low_noise_data.taxa
    for sub in low_noise_data:
        #print(sub)
        latent_y = latent_data._data[sub.name]
        low_y = sub.matrix()["abs"]
        med_y = med_noise_data[sub.name].matrix()["abs"]
        high_y = high_noise_data[sub.name].matrix()["abs"]
        savepath = path / "seed_{}".format(seed) / "combined_plot" / "{}".format(sub.name)
        savepath.mkdir(parents=True, exist_ok=True)
        for i in range(low_y.shape[0]):
            taxa = all_taxa[i]
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_xlabel("Days")
            ax.set_ylabel("CFUs/g")
            ax.set_title("Mouse {}, {}".format(sub.name, md2.pylab.base.taxaname_for_paper(
                taxa, all_taxa)))
            ax.plot(sub.times, latent_y[i], label="latent state")
            ax.plot(sub.times, low_y[i], label="low noise")
            ax.plot(sub.times, med_y[i], label="medium noise")
            ax.plot(sub.times, high_y[i], label="high noise")
            ax.legend()
            ax.set_yscale("log")
            fig.savefig(savepath / "{}.pdf".format(taxa.name))
def main():

    args = parse_arguments()
    low_noise_params = {"a0":args.negbin_a0_low, "a1":args.negbin_a1_low,
      "scale":args.noise_scale_low, "depth":args.read_depth}
    medium_noise_params = {"a0":args.negbin_a0_med, "a1":args.negbin_a1_med,
      "scale":args.noise_scale_medium, "depth":args.read_depth}
    high_noise_params = {"a0":args.negbin_a0_high, "a1":args.negbin_a1_high,
      "scale":args.noise_scale_high, "depth":args.read_depth}

    print("Generating data for seed: {} and exporting results to {}".format(
        args.seed, args.save_path))
    mcmc = md2.BaseMCMC.load(Path(args.mcmc_file_loc) / "mcmc.pkl")
    semi_syn = generate_semi_synthetic_trajectory(mcmc, args.seed,
        "semi-synthetic-{}".format(args.seed), args.dt, args.process_var,
        n_subjects=4, path=Path(args.save_path), plot=True)

    low_semi_synthetic_study = add_measurement_noise(semi_syn, args.seed,
        low_noise_params, "low", "semi_synthetic_low", Path(args.save_path))
    med_semi_synthetic_study = add_measurement_noise(semi_syn, args.seed,
        medium_noise_params, "medium", "semi_synthetic_medium",Path(args.save_path))
    high_semi_synthetic_study = add_measurement_noise(semi_syn, args.seed,
        high_noise_params, "high", "semi_synthetic_high",Path(args.save_path))

    combined_plot(semi_syn, low_semi_synthetic_study, med_semi_synthetic_study,
       high_semi_synthetic_study, args.seed, Path(args.save_path))

main()
