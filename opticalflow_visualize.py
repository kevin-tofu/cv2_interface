
def get_timeseries(args):
    import os, json

    fname_list = os.listdir(args.dir_data)
    data_list = list()

    for f in fname_list:
        fpath = f"{args.dir_data}{f}"
        fname_base = os.path.splitext(os.path.basename(f))[0]

        print(fpath)
        with open(fpath) as f:
            df = json.load(f)
            data_list.append({"fname": fname_base, "flow_timeseries": df["flow_time"]})

    return data_list

def visualize_flow_sum(args):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import pylab as plt

    data_list = get_timeseries(args)

    fname_list = list()
    plt.figure()
    for data in data_list:
        fname_list.append(data["fname"])
        plt.plot(data["flow_timeseries"])
    plt.logend(fname_list)
    plt.savefig(f"{args.dir_data}flow_sum.png")
