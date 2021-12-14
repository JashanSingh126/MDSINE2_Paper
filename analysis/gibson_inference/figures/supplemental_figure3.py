import numpy as np
import matplotlib.pyplot as plt
import mdsine2 as md2
import argparse
import os
from matplotlib import rcParams
from matplotlib import font_manager

rcParams['pdf.fonttype'] = 42

font_dirs = ['gibson_inference/figures/arial_fonts']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
#print(font_files)

for font_file in font_files:
    #print(font_file)
    ff = font_file.split("/")[-1]
    if "._" not in ff:
        font_manager.fontManager.addfont(font_file)

# change font
rcParams['font.family'] = 'Arial'


def filter(study, dtype, threshold, min_num_consecutive, min_num_subjects,
	colonization_time, max_n_otus=None):

    study = md2.consistency_filtering(subjset=study, dtype=dtype,
        threshold=threshold, min_num_consecutive=min_num_consecutive,
        min_num_subjects=min_num_subjects, colonization_time=colonization_time)
    if max_n_otus is not None:
        n = args.max_n_otus
        if n <= 0:
            raise ValueError('`max_n_otus` ({}) must be > 0'.format(n))
        to_delete = []
        for taxon in study.taxa:
            if taxon.idx >= n:
                to_delete.append(taxon.name)
        study.pop_taxa(to_delete)
    #print('{} taxa left in {}, n_days = {}'.format(len(study.taxa), study.name))
    return len(study.taxa)

def run_and_save_filtering_results(study_healthy, study_uc):
    fig = plt.figure(figsize=(15, 20))
    axes_set = {}
    i = 1
    for subjs in study_healthy:
        axes_set[i] = fig.add_subplot(4, 3, i)
        i += 1

    i = 7
    for subjs in study_uc:
        axes_set[i] = fig.add_subplot(4, 3, i)
        i += 1

    make_plot(axes_set, study_healthy, 1, ["A", "B", "C", "D"])
    make_plot(axes_set, study_uc, 7, ["E", "F", "G", "H", "I"])
    fig.subplots_adjust(hspace=0.3)
    loc = "gibson_inference/figures/output_figures/"
    os.makedirs(loc, exist_ok=True)
    fig.subplots_adjust(wspace=0.25)
    fig.savefig(loc + "supplemental_figure3.pdf", bbox_inches="tight")


def make_plot(axes_set, study, start, title_li):

    #days = [7]
    days = [1, 2, 3, 4, 5, 6, 7]
    threshold_values = np.linspace(0, 0.0005, 11)
    n_subjects = len(study)

    max_n = 0
    min_n = np.infty
    used_keys = []
    col_time=5
    i = 1
    study_name = "Healthy"
    if study.name != "healthy":
        study_name = "Dysbiotic"
    for subj_num in range(start, n_subjects+start):
        print("subj number:", subj_num)
        axes = axes_set[subj_num]
        used_keys.append(subj_num)
        axes.set_xlabel("Minimum Rel Abundance", fontweight="bold",
            fontsize=15)
        axes.set_ylabel("# OTUs remaining", fontsize=15, fontweight="bold")
        if i ==1:
            axes.set_title("{}, {} mouse".format(study_name,
                i), fontsize=18, fontweight="bold", loc="center")
        else:
            axes.set_title("{}, {} mice".format(study_name,
                i), fontsize=18, fontweight="bold", loc="center")
        #axes.text(0, 1.1, title_li[i-1], fontweight="bold", fontsize=18, transform = axes.transAxes)
        axes.set_title(title_li[i-1], fontweight="bold", fontsize=18, loc="left")
        axes.ticklabel_format(axis="x", style="sci")
        used_n = 0
        for d in days:
            results_n = [np.nan]
            for t in threshold_values[1:]:
                n_taxa = filter(study, "rel", t, d, i, col_time)
                max_n = max(n_taxa, max_n)
                min_n = min(n_taxa, min_n)
                results_n.append(n_taxa)
                if t == 0.0001 and d==7:
                    if subj_num==2 or subj_num==8:
                        used_n = n_taxa
                        print("used:", used_n)
                #print("subjs:{}, days:{}, n_taxa:{}, thresholds:{}".format(
                #i, d, n_taxa, t))
            axes.plot(threshold_values, results_n, label="{} consecutive".format(d))
        if study.name=="healthy" and subj_num==3:
            axes.legend(bbox_to_anchor=(1.04, 0), loc=2, fontsize=13,
            title="$\\bf{\# timepoints}$", title_fontsize=15)
        if subj_num==2 or subj_num==8 and used_n !=0:
            x = np.linspace(0, 0.0001, 11)
            #y = np.linspace(used_n, max_n + 20, 50)
            y = np.linspace(0, used_n, 50)
            axes.plot(x, [used_n] * x.shape[0], "--", color="red")
            axes.plot([0.0001] * y.shape[0], y, "--", color="red")
            axes.scatter(0.0001, used_n, color="red")
        axes.tick_params(which="both", labelsize=13)
        i += 1

    for axes_key in axes_set:
        axes = axes_set[axes_key]
        if axes_key in used_keys:
            axes.set_xlim(0, 0.00052)
            axes.set_ylim(min_n-20, max_n+20)
            axes.grid()

def parse_args():

    parser = argparse.ArgumentParser(description = "files needed for making"\
    "supplemental figure 3")
    parser.add_argument("-file1", "--healthy_study", required = "True",
       help = "pl.base.Study for Healthy data in pkl format")
    parser.add_argument("-file2", "--uc_study", required = "True",
       help = ".pl.base.Study for UC data in pkl format")


    return parser.parse_args()

if __name__ =="__main__":

    print("Making Supplemental Figure 3")
    args = parse_args()

    healthy_study = md2.Study.load(args.healthy_study)
    uc_study = md2.Study.load(args.uc_study)

    run_and_save_filtering_results(healthy_study, uc_study)
    print("Done Making Supplemental Figure 3")
