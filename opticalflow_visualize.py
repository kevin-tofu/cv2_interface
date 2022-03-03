
def get_fpath(_dir, ext='json'):
    import os, json, glob
    # fname_list = os.listdir(args.dir_data)
    fpath_list  = glob.glob(f"{_dir}/*.{ext}")
    return fpath_list

def get_timeseries(args):


    import os, json

    # fname_list = os.listdir(args.dir_data)
    fname_list = get_fpath(args.dir_data, "json")
    data_list = list()

    for f in fname_list:
        fpath = f
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
    plt.legend(fname_list)
    plt.savefig(f"{args.dir_data}flow_sum.png")


def test(args):

    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import pylab as plt

    fpath_list = get_fpath(args.dir_data, "mp4")

    data_list_list = list()

    for fpath in fpath_list:
        print(fpath)

        cap = cv2.VideoCapture(fpath)
        if cap.isOpened() == False:
            cap.open(fpath)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        hight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        data_list = list()
        while(1):
            ret, rgb = cap.read()
            if ret == False:
                break
                
            hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
            hsv_np = np.asarray(hsv)
            # hsv_np[hsv_np > 5] = 0

            data_list.append(np.sum(hsv_np[..., 2]))

        data_list_list.append(data_list)
    cap.release()

    plt.figure()
    for data in data_list_list:
        plt.plot(data)
    
    plt.grid()
    # plt.yscale('log')
    plt.savefig(f"{args.dir_data}flow_sum2.png")
