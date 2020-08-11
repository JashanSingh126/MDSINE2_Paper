
import sys
import pandas as pd
import numpy as np
from scipy.integrate import ode

def well_mixed_glv(t, y, growth, interaction):
    Y = y
    dydt = []
    for i in range(0, len(Y)):
        d_gr = growth[i] * Y[i]
        d_int = 0
        for j in range(0, len(Y)):
            d_int = d_int + interaction[i,j]*Y[i]*Y[j]
        dydt.append(d_gr + d_int)
    return dydt
    

def integrate(myode, tf, dt, out=None):
    while myode.successful() and myode.t < tf:
        out.append(myode.integrate(myode.t + dt))
    return out


def make_solver(y0, bf_args):
    myode = ode(well_mixed_glv)
    growth = bf_args[:, 0]
    print(growth)
    interactions = bf_args[:, 1:]
    print(interactions)
    myode.set_f_params(growth, interactions)
    myode.set_initial_value(y0)
    myode.set_integrator('dop853', nsteps=5000)  # RK seems more stable than vode
    # myode.set_integrator('vode', nsteps=50000)  # RK seems more stable than vode
    return myode


def sim_n(filename):
    output_file = filename + '.out'
    #data = np.loadtxt(filename, delimiter=',')
    data = pd.read_csv(filename, header=None)
    print(data)
    data2 = np.asarray(data.iloc[:,1:])
    y0 = data2[:, 0]
    glv_pars = data2[:, 1:]  # split in make_solver
    print(y0)
    end_time = 200
    dt = 2  # for reported times, not for integration!

    myo = make_solver(y0, glv_pars)
    #exit()
    y = integrate(myo, end_time, dt, [])

    # concatenate and write to file
    np_time = np.arange(0, end_time, step=dt)
    np_y = np.asarray(y)
    out_time_y = np.vstack((np_time, np_y.T))
    pd_out_time_y = pd.DataFrame(out_time_y)
    snames = data[data.columns[0]].tolist()
    snames = ['Time'] + snames
    print(snames)
    #exit()
    pd_out_time_y.index = snames
    export_csv = pd_out_time_y.to_csv(output_file,header=False)



if __name__ == '__main__':
    for i in range(1, len(sys.argv)):
        sim_n(sys.argv[i])
