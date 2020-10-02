import numpy as np
import pickle as pkl
import os

from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

sample_id_to_subject_id = {}
subject_id_time = {}
subject_id_u = {}

def plot_trajectories(Y, T, output_dir, outfile):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    def plot_bar(ax, y, time, unique_color_id, remaining_ids):
        T = y.shape[0]
        cm = plt.get_cmap("tab20c")
        colors = [cm(i) for i in range(20)]
        #time = np.array([t for t in range(T)])
        widths = np.concatenate((time[1:] - time[:-1], [1])).astype(float)
        widths[widths > 1] = 1

        widths -= 1e-1

        y_colors = y[:,unique_color_id]
        ax.bar(time, y_colors[:,0], width=widths, color=colors[0], align="edge")
        for j in range(1, y_colors.shape[1]):
            ax.bar(time, y_colors[:,j], bottom=y_colors[:,:j].sum(axis=1), width=widths, color=colors[j], align="edge")
        
        ax.bar(time, y[:,remaining_ids].sum(axis=1), bottom=y_colors.sum(axis=1), width=widths, color=colors[19], align="edge")
        #ax.set_title("Relative Abundances", fontsize=10)
        #ax.legend(prop={"size" : 4}, bbox_to_anchor=[-0.1,1.225], loc="upper left", ncol=4)

    def find_top_ids(Y, n):
        ntaxa = Y[0].shape[1]
        rel_abun = np.zeros(ntaxa)
        for y in Y:
            tpts = y.shape[0]
            denom = y.sum(axis=1,keepdims=True)
            denom[denom == 0] = 1
            p = y / denom
            rel_abun += p.sum(axis=0) / tpts
        ids = np.argsort(-rel_abun)
        return np.sort(ids[:n]), np.sort(ids[n:])

    N = len(Y)
    top19_ids, remaining_ids = find_top_ids(Y, 19)
    fig, ax = plt.subplots(nrows=N,ncols=1,figsize=(N,2*N))
    for i in range(N):
        denom = Y[i].sum(axis=1)
        denom[denom == 0] = 1
        plot_bar(ax[i], (Y[i].T / denom).T, T[i], top19_ids, remaining_ids)


    outfile = os.path.splitext(outfile)[0]
    plt.tight_layout()
    plt.savefig(output_dir + "/" + outfile + ".pdf")
    plt.close()


def compute_breakpoints(effects):
    """Break the spline when external perturbations occur."""
    breakpoints = []
    for u in effects:
        in_perturb = False
        v = []
        for i,ut in enumerate(u):
            if i == 0 or i == u.shape[0]-1:
                v.append(1)
                continue
            if np.any(ut) > 0 and not in_perturb:
                v.append(1)
                in_perturb = True
            elif np.any(ut) > 0 and in_perturb:
                v.append(0)
            elif np.all(ut) == 0 and in_perturb:
                i = 1 if v[i-1] == 0 else 0
                v.append(i)
                in_perturb = False
            else:
                v.append(0)
        v = np.array(v)
        breakpoints.append(np.nonzero(v)[0])
    return breakpoints


def denoise(counts, t_pts, effects=None):
    """Takes a sequence of counts at t_pts, and returns denoised estimates
    of latent trajectories."""
    ntaxa = counts[0].shape[1]
    denoised_traj = []

    if effects is not None:
        breakpoints = compute_breakpoints(effects)
        for c,t,b in zip(counts,t_pts,breakpoints):
            denoised = np.zeros(c.shape)
            mass = c.sum(axis=1,keepdims=True)
            p = c / c.sum(axis=1,keepdims=True)
            p[p==0] = 1e-5
            p /= p.sum(axis=1,keepdims=True)
            c = (mass.T*p.T).T
            for i in range(ntaxa):
                for j in range(1,b.size):
                    start = b[j-1]
                    end = b[j]+1
                    k = 5 if end - start <= 3 else 5
                    #k = 3
                    f = UnivariateSpline(t[start:end],c[start:end,i],k=k)
                    denoised[start:end,i] = f(t[start:end])
            denoised[0] = c[0]
            denoised = np.clip(denoised, np.min(denoised[denoised > 0]), np.inf)
            denoised_traj.append(denoised)
    else:
        for c,t in zip(counts,t_pts):
            denoised = np.zeros(c.shape)
            k = 3 if t.shape[0] <= 5 else 5
            for i in range(ntaxa):
                f = UnivariateSpline(t,c[:,i],k=k)
                denoised[:,i] = f(t)
            denoised = np.clip(denoised, np.min(denoised[denoised > 0]), np.inf)
            denoised /= denoised.sum(axis=1,keepdims=True)
            denoised_traj.append(denoised)

    return denoised_traj


with open("data_uc/metadata.txt", "r") as f:
    for line in f:
        line = line.split()

        if "sampleID" in line[0]:
            continue

        sample_id = line[0]
        subject_id = line[2]
        day = float(line[3])
        perturb = float(line[4])

        sample_id_to_subject_id[sample_id] = subject_id
        subject_id_time[subject_id] = subject_id_time.get(subject_id, []) + [day]
        subject_id_u[subject_id] = subject_id_u.get(subject_id, []) + [perturb]


counts = np.loadtxt("data_uc/counts.txt", delimiter="\t", dtype=str, comments="!")
names = counts[1:, 0]
counts = counts[:, 1:]
# print(counts)
# print(names)

subject_id_counts = {}

for row in counts.T:
    # print(row)
    # print(len(row))
    sample_id = row[0]
    counts = row[1:].astype(float)
    subject_id = sample_id_to_subject_id[sample_id]

    counts /= 1000
    if subject_id in subject_id_counts:
        subject_id_counts[subject_id] = np.vstack( (subject_id_counts[subject_id], np.array(counts)) )
    else:
        subject_id_counts[subject_id] = np.array(counts)


Y_uc = []
U_uc = []
T_uc = []
zero_counts = 0
total_counts = 0
for subject_id in sorted(subject_id_counts):
    y = np.array(subject_id_counts[subject_id])
    t = np.array(subject_id_time[subject_id])
    u = np.array(subject_id_u[subject_id])
    u = u.reshape((u.size, 1))
    zero_counts += y[y == 0].size
    total_counts += y.size

    Y_uc.append(y)
    U_uc.append(u)
    T_uc.append(t)


Y_uc_denoised = denoise(Y_uc, T_uc, effects=U_uc)

plot_trajectories(Y_uc_denoised, T_uc, "./", "uc-denoised")
plot_trajectories(Y_uc, T_uc, "./", "uc-raw")

pkl.dump(Y_uc, open("Y_uc.pkl", "wb"))
pkl.dump(Y_uc_denoised, open("Y_uc_denoised.pkl", "wb"))
pkl.dump(U_uc, open("U_uc.pkl", "wb"))
pkl.dump(T_uc, open("T_uc.pkl", "wb"))

print("sample size", len(Y_uc))
print("% zero", zero_counts / total_counts)