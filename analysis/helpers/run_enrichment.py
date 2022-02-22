import enrichment as enrich
import argparse
import mdsine2 as md2

def parse_args():

    parser = argparse.ArgumentParser(description = "files needed for making"\
    "supplemental figure 5")
    parser.add_argument("-m_loc", "--mcmc_loc", required = "True",
       help = "a pl.BaseMCMC pkl file for healthy runs")
    parser.add_argument("-l", "--level", required="True",
        help="the level at which enrichment analysis is performed")
    parser.add_argument("-o_loc", "--output_loc", required="True",
        help = "directory(folder name) where the output is saved")
    parser.add_argument("-o_name", "--output_name", required="True",
        help = "name of the output")

    return parser.parse_args()


if __name__ =="__main__":

    args = parse_args()
    mcmc = md2.BaseMCMC.load(args.mcmc_loc)

    df_enrich = enrich.run_enrichment_level(mcmc, args.level,
        "enrichment_test")
    df_members = enrich.pivot_cluster_membership(mcmc, args.level)
    
    enrich.simple_plot(df_enrich, df_members, mcmc.graph.data.subjects.name, args.level,
       args.output_loc, args.output_name)
