import numpy as np
import matplotlib.pyplot as plt
import mdsine2 as md2
import argparse
import os 


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

def run_and_save_filtering_results(study):

	fig = ""
	col_time = 5
	axes_set = {}
	i = 1
	if study.name =="healthy":
		fig = plt.figure(figsize=(12, 9))
		fig.text(0.47, 0.95, "Healthy", fontsize=20, fontweight="bold")
		for subjs in study:
			axes_set[i] = fig.add_subplot(2, 2, i)
			i += 1
	else:
		fig = plt.figure(figsize=(18, 9))
		fig.text(0.45, 0.95, "Ulcerative Colitis", fontsize=20, fontweight="bold")
		for subjs in study:
			axes_set[i] = fig.add_subplot(2, 3, i)
			i += 1

	#days = [7]
	days = [1, 2, 3, 4, 5, 6, 7]
	threshold_values = np.linspace(0, 0.0005, 11)
	n_subjects = len(study)
	print("n subjs:", n_subjects, i)
	max_n = 0
	min_n = np.infty

	for subj_num in range(1, n_subjects+1):
		print("subj bumber:", subj_num)
		axes = axes_set[subj_num]
		axes.set_xlabel("Minimum Relative Abundance")
		axes.set_ylabel("# OTUs remaining")
		axes.set_title("Subject {}".format(subj_num))

		used_n = 0

		for d in days:
			results_n = [np.nan]
			for t in threshold_values[1:]:
				n_taxa = filter(study, "rel", t, d, subj_num, col_time)
				max_n = max(n_taxa, max_n)
				min_n = min(n_taxa, min_n)
				results_n.append(n_taxa)
				if t == 0.0001 and d==7 and subj_num==2:
					used_n = n_taxa
					print("used:", used_n)
				print("subjs:{}, days:{}, n_taxa:{}, thresholds:{}".format(
					subj_num, d, n_taxa, t))
			axes.plot(threshold_values, results_n, label="{} consecutive".format(d))

		if study.name=="healthy" and subj_num==2:
			axes.legend(bbox_to_anchor=(1.04, 0), loc=2)
		if study.name != "healthy":
			if subj_num==3:
				axes.legend(bbox_to_anchor=(1.04, 0), loc=2)

		if subj_num==2 and used_n !=0:
			print("used parameters")
			x = np.linspace(0, 0.0001, 11)
			#y = np.linspace(used_n, max_n + 20, 50)
			y = np.linspace(0, used_n, 50)
			axes.plot(x, [used_n] * x.shape[0], "--", color="red")
			axes.plot([0.0001] * y.shape[0], y, "--", color="red")
			axes.scatter(0.0001, used_n, color="red")

	for axes_key in axes_set:
		axes = axes_set[axes_key]
		axes.set_xlim(0, 0.00052)
		axes.set_ylim(min_n-20, max_n+20)
		axes.grid()



	fig.subplots_adjust(hspace=0.3)
	loc = "gibson_inference/figures/output_figures/"
	os.makedirs(loc, exist_ok=True)
	fig.savefig(loc + "supplemental_figure2_{}.pdf".format(study.name), bbox_inches="tight")


def parse_args():

    parser = argparse.ArgumentParser(description = "files needed for making"\
    "supplemental figure 2")
    parser.add_argument("-file1", "--healthy_study", required = "True",
       help = "pl.base.Study for Healthy data in pkl format")
    parser.add_argument("-file2", "--uc_study", required = "True",
       help = ".pl.base.Study for UC data in pkl format")


    return parser.parse_args()

if __name__ =="__main__":

	args = parse_args()

	healthy_study = md2.Study.load(args.healthy_study)
	uc_study = md2.Study.load(args.uc_study)

	run_and_save_filtering_results(healthy_study)
	run_and_save_filtering_results(uc_study)
