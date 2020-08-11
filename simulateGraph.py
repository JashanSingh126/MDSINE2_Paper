import networkx as nx
import numpy as np
import torch
import SampleUtils
import matplotlib.pyplot as plt

class MicrobiomeGraph():
    def __init__(self,num_nodes):
        self.num_nodes = num_nodes
        self.connectMatrix = np.zeros((self.num_nodes,self.num_nodes))
        self.baselineNodeIntensity = np.zeros(self.num_nodes)
        self.perturbedNodeIntensity = np.zeros(self.num_nodes)
        #self.labels = torch.zeros(self.num_nodes,dtype=torch.bool)
        self.graph = nx.Graph()
        self.similarity_graph = nx.Graph()
        self.cov = []

        # probability of edge between any pair of nodes
        self.prob_edge = 0.01

        # assume edge lengths arise from a truncated normal
        self.edge_length_mean = 1.0
        self.edge_length_std = 0.5

        # assume baseline values for nodes are drawn from truncated normal
        self.baseline_value_mean = 10.0
        self.baseline_value_std = 3.0

        # number of 'centers' in the graph generating a perturbation
        self.num_centers = 4

        # assume the radius of 'influence' is sampled from a truncated normal
        self.center_rad_mean = 10.0
        self.center_rad_std = 2.0

        # parameter for probability distribution on spreading influence of centers on other nodes
        self.center_rad_scale = 4.0

        # mean and std of size of perturbation
        self.center_value_mean = 10.0
        self.center_value_std = 4.0

        # variability in values per subject
        self.subject_std = 0.5
        self.subject_perturb_std = 1.5

    def simTopology(self):
        for i in range(self.num_nodes):
            self.graph.add_node(str(i+1))
            self.similarity_graph.add_node(str(i+1))

        # generate topology
        for i in range(self.num_nodes-1):
            for j in range(i,self.num_nodes):
                u = np.random.uniform()
                if u <= self.prob_edge:
                    ew = SampleUtils.sample_trunc_normal(self.edge_length_mean,self.edge_length_std,0,np.inf)
                    self.connectMatrix[i,j] = ew
                    self.graph.add_edge(str(i+1), str(j+1), weight=ew)
                    self.similarity_graph.add_edge(str(i+1), str(j+1), weight=1.0/ew)

    def simBaselineNodeValues(self):
        cov = nx.laplacian_matrix(self.similarity_graph).todense()
        self.cov = np.linalg.pinv(cov)
        mu = np.ones(self.num_nodes) * self.baseline_value_mean
        self.baselineNodeIntensity = np.random.multivariate_normal(mu, self.cov*np.power(self.baseline_value_std,2.0))

    def simPerturbation(self):
        # probability threshold for propogating perturbation from center
        thresh = 0.5

        for i in range(self.num_centers):
            r = SampleUtils.sample_trunc_normal(self.center_rad_mean,self.center_rad_std,0,np.inf)
            nim = SampleUtils.sample_trunc_normal(self.center_value_mean,self.center_value_std,0,np.inf)
            for j in range(self.num_nodes):
                try:
                    d = nx.dijkstra_path_length(self.graph,str(i+1),str(j+1),'weight')
                except nx.exception.NetworkXNoPath:
                    d = np.inf

                p = np.exp(-d*self.center_rad_scale/r)

                if p >= thresh:
                    # self.perturbedNodeIntensity[j] = nim + pl.random.truncnormal.sample(mean=, std=, low=, high=)
                    self.perturbedNodeIntensity[j] = nim + SampleUtils.sample_trunc_normal(0,0.1,0,np.inf)

        #for i in range(self.num_nodes):
        #    self.graph.nodes[str(i+1)]['intensity'] = self.baselineNodeIntensity[i] + self.perturbedNodeIntensity[i]

    def sampleSubjectIntensities(self,do_perturb):
        # assume correlated noise according to graph structure
        pss = self.baselineNodeIntensity + np.random.multivariate_normal(np.zeros(self.num_nodes), self.cov*np.power(self.subject_std,2.0))

        if do_perturb:
            for i in range(self.num_nodes):
                if self.perturbedNodeIntensity[i] > 0.0:
                    pss[i] += SampleUtils.sample_trunc_normal(self.perturbedNodeIntensity[i],np.power(self.subject_perturb_std,2.0),0,np.inf)

        return pss

    def drawGraph(self,vals):
        pos = nx.kamada_kawai_layout(self.graph)

        active_centers = ['1','2','3','4']
        nx.draw_networkx_nodes(self.similarity_graph, pos, size=300, nodelist=active_centers, node_shape='s',node_color='b')

        intensities = list()
        #for key, value in nx.get_node_attributes(self.graph,'intensity').items():
        #    intensities.append(value)
        for v in vals:
            intensities.append(v)

        nx.draw_networkx_nodes(self.graph, pos, size=300, cmap=plt.get_cmap('inferno'), node_color=intensities)

        labels = nx.get_edge_attributes(self.similarity_graph,'weight')
        rl = list()
        for key, value in labels.items():
            rl.append(np.round(0.5/value)+1)

        nx.draw_networkx_edges(self.similarity_graph,pos,width=rl)

        plt.axis('off')
        plt.show()

mg = MicrobiomeGraph(100)
mg.simTopology()
mg.simBaselineNodeValues()
mg.simPerturbation()

mg.drawGraph(mg.sampleSubjectIntensities(True))
mg.drawGraph(mg.sampleSubjectIntensities(True))
#mg.drawGraph(mg.baselineNodeIntensity)
#mg.drawGraph(mg.baselineNodeIntensity+mg.perturbedNodeIntensity)
