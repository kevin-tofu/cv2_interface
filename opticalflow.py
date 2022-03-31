import numpy as np
import cv2
import os
"""
http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
https://qiita.com/icoxfog417/items/357e6e495b7a40da14d8
https://www.cresco.co.jp/blog/entry/16891/
https://www.programcreek.com/python/example/89313/cv2.calcOpticalFlowFarneback
https://stackoverflow.com/questions/41661517/drawing-results-of-calcopticalflowfarneback-in-opencv-without-loop

"""


def get_keywords_opticalflow(**kwargs):

    pyr_scale = kwargs['_pyr_scale'] if '_pyr_scale' in kwargs else 0.5
    levels = kwargs['_levels'] if '_levels' in kwargs else 3
    winsize = kwargs['_winsize'] if '_winsize' in kwargs else 15
    iterations = kwargs['_iterations'] if '_iterations' in kwargs else 3
    poly_n = kwargs['_poly_n'] if '_poly_n' in kwargs else 5
    poly_sigma = kwargs['_poly_sigma'] if '_poly_sigma' in kwargs else 1.2
    flags = kwargs['_flags'] if '_flags' in kwargs else 0

    return pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags


def get_keywords_lktrack(**kwargs):

    max_corners = kwargs['_max_corners'] if '_max_corners' in kwargs else 100
    quality_level = kwargs['_quality_level'] if '_quality_level' in kwargs else 0.3
    min_distance = kwargs['_min_distance'] if '_min_distance' in kwargs else 7.0
    blocksize = kwargs['_blocksize'] if '_blocksize' in kwargs else 7
    win_size = kwargs['_win_size'] if '_win_size' in kwargs else 15
    max_level = kwargs['_max_level'] if '_max_level' in kwargs else 2
    lk_criteria = kwargs['_lk_criteria'] if '_lk_criteria' in kwargs else cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT
    lk_count = kwargs['_lk_count'] if '_lk_count' in kwargs else 30
    epsilon = kwargs['_epsilon'] if '_epsilon' in kwargs else 0.03
    # test = kwargs['test'] 
    feature_params = dict(maxCorners=max_corners, 
                            qualityLevel=quality_level,
                            minDistance=min_distance,
                            blockSize=blocksize
                        )

    lk_params = dict(winSize=(win_size, win_size), 
                        maxLevel=max_level,
                        criteria=(lk_criteria, lk_count, epsilon)
                    )
    return feature_params, lk_params



def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    # vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = img

    cv2.polylines(vis, lines, 0, (0, 255, 0), thickness=4)
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    
    return vis


def opticalflow_dense_image_base(img_prev_gray,\
                                 img_next_gray,\
                                 **kwargs):
    """
    """
    pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags = get_keywords_opticalflow(**kwargs)
    flow = cv2.calcOpticalFlowFarneback(img_prev_gray, img_next_gray, \
                                        None, pyr_scale, levels, winsize, \
                                        iterations, poly_n, poly_sigma, flags)
    
    
    # print(flow)
    # flow = 0
    return flow


def opticalflow_dense_image(fpath_img_prev,\
                            fpath_img_next,\
                            **kwargs):
    """
    """

    img_prev = cv2.imread(fpath_img_prev)
    img_next = cv2.imread(fpath_img_next)

    img_prev_gray = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    img_next_gray = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
    flow = opticalflow_dense_image_base(img_prev_gray,\
                                        img_next_gray,\
                                        **kwargs)
    # flow_mag = np.asarray(mag)
    flow_list = np.asarray(flow)
    flow_list_r = np.round(flow_list, decimals=5).tolist()

    return flow_list_r



def opticalflow_dense_image_draw_base(img_prev,\
                                      img_next,\
                                      export='org+flow',\
                                      **kwargs):
    """
    """

    # export = kwargs['export']
    img_prev_gray = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    img_next_gray = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
    flow = opticalflow_dense_image_base(img_prev_gray,\
                                        img_next_gray,\
                                        **kwargs)
    hsv = np.zeros_like(img_next)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    

    if export == 'org-flow':
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img_ret = cv2.vconcat([rgb, img_next_gray])

    elif export == 'org+flow':

        if "_draw_step" in kwargs:
            draw_step = kwargs['_draw_step']
        else:
            draw_step = 32

        # print("draw_step: ", draw_step)
        img_ret = draw_flow(img_next, flow, step=draw_step)

    elif export == 'flow':
        # hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = mag
        img_ret = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        img_ret = {}
    
    return img_ret


def opticalflow_dense_image_draw(fpath_img_prev: str,\
                                 fpath_img_next: str,\
                                 fpath_ex: str, \
                                 export: str,\
                                 **kwargs):

    # export = kwargs['export']
    img_prev = cv2.imread(fpath_img_prev)
    img_next = cv2.imread(fpath_img_next)
    img_optflow = opticalflow_dense_image_draw_base(img_prev,\
                                                    img_next,\
                                                    export,\
                                                    **kwargs)

    
    cv2.imwrite(fpath_ex, img_optflow)



def opticalflow_dense_sum_video(fpath, **kwargs):
        """
        """
        import math

        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags = get_keywords_opticalflow(**kwargs)

        cap = cv2.VideoCapture(fpath)
        # pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags = 0.5, 3, 15, 3, 5, 1.2, 0

        k = cap.isOpened()
        if k == False:
            cap.open(fpath)


        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        flow_list = list()
        while(1):
            ret, frame2 = cap.read()
            if ret == False:
                break

            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
            
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_mag = np.asarray(mag).astype(np.float64)
            flow_mag[flow_mag==np.inf] = 0
            flow_mag_sum = float(np.sum(flow_mag))
            flow_mag_sum = round(flow_mag_sum, 3)
            # print(flow_mag_sum, type(flow_mag_sum), flow_mag.shape, flow_mag.max(), flow_mag.min())

            if math.inf == flow_mag_sum:
                flow_mag_sum = -1
            flow_list.append(flow_mag_sum)
            

            # mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            # hsv[...,0] = ang*180/np.pi/2
            # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            # rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

            # cv2.imshow('frame2',rgb)
            # k = cv2.waitKey(30) & 0xff
            # if k == 27:
            #     break
            # elif k == ord('s'):
            #     cv2.imwrite('opticalfb.png',frame2)
            #     cv2.imwrite('opticalhsv.png',rgb)
            prvs = next

        cap.release()
        # cv2.destroyAllWindows()
        
        return flow_list




# def set_audio(path_src, path_img, path_dst):
def set_audio(path_src, path_dst):
    """
    https://kp-ft.com/684
    https://stackoverflow.com/questions/46864915/python-add-audio-to-video-opencv
    """

    import os, shutil
    import moviepy.editor as mp
    import time

    root_ext_pair = os.path.splitext(path_src)
    # path_audio = f"{root_ext_pair[0]}-audio.mp3"
    path_dst_copy = f"{root_ext_pair[0]}-copy{root_ext_pair[1]}"
    shutil.copyfile(path_dst, path_dst_copy)
    time.sleep(0.5)

    # Extract audio from input video.                                                                     
    clip_input = mp.VideoFileClip(path_src)
    # clip_input.audio.write_audiofile(path_audio)
    # Add audio to output video.                                                                          
    clip = mp.VideoFileClip(path_dst_copy)
    clip.audio = clip_input.audio

    time.sleep(0.5)
    clip.write_videofile(path_dst)

    time.sleep(0.5)
    os.remove(path_dst_copy)


def opticalflow_dense_draw(fpath: str, \
                           fpath_dst: str, \
                           export: str, \
                           **kwargs):
        """
        """
        
        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags = get_keywords_opticalflow(**kwargs)
        # export = kwargs['export']
        if "_draw_step" in kwargs:
            draw_step = kwargs['_draw_step']
        else:
            draw_step = 32
        # print("draw_step: ", draw_step)

        cap = cv2.VideoCapture(fpath)
        k = cap.isOpened()
        if k == False:
            cap.open(fpath)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # print(fpath_ex)
        fpath_ex = os.path.splitext(fpath_dst)[-1]
        if fpath_ex == ".mp4" or fpath_ex == ".MP4":
            fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        else:
            raise ValueError("file extention error")

        if export == 'org-flow':
            writer = cv2.VideoWriter(fpath_dst, fmt, fps, (width, 2*height))
        elif export == 'org+flow':
            writer = cv2.VideoWriter(fpath_dst, fmt, fps, (width, height))
        elif export == 'flow':
            writer = cv2.VideoWriter(fpath_dst, fmt, fps, (width, height))

        # player = MediaPlayer(fpath)

        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        # flow_list = list()
        # while(1):
        for i in range(frame_count):

            ret, frame2 = cap.read()
            if ret == False:
                break

            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
            # flow_list.append(flow.tolist())

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[...,0] = ang*180/np.pi/2
            

            if export == 'org-flow':
                hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                rgb2 = cv2.vconcat([rgb, frame2])
                writer.write(rgb2)

            elif export == 'org+flow':
                

                # print(draw_step)
                rgb2 = draw_flow(frame2, flow, step = draw_step)
                writer.write(rgb2)

            elif export == 'flow':
                # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                writer.write(rgb)
                # writer.write(flow)

            # cv2.imshow('frame2',rgb)
            # k = cv2.waitKey(30) & 0xff
            # if k == 27:
            #     break
            # elif k == ord('s'):
            #     cv2.imwrite('opticalfb.png',frame2)
            #     cv2.imwrite('opticalhsv.png',rgb)
            prvs = next

        cap.release()
        if export in ['org-flow', 'org+flow', 'flow']:
            writer.release()

            set_audio(fpath, fpath_dst)
            # set_audio2(fpath, fpath_ex)
        


def lk_track(fpath, **kwargs):

    feature_params, lk_params = get_keywords_lktrack(**kwargs)
    cap = cv2.VideoCapture(fpath)

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    # mask = np.zeros_like(old_frame)

    data = list()
    frame_id = 0


    while(1):
        ret, frame = cap.read()
        if ret == False:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        if len(good_new) == 0:
            continue
        
        flow_r = np.round(good_new - good_old, decimals=4)
        good_new_r = np.round(good_new, decimals=4)

        # print(flow_r)
        # print(good_new_r)

        data.append(dict(frame=frame_id, flow=flow_r.tolist(), 
                         position=good_new_r.tolist()
        ))


        # draw the tracks
        # for i,(new,old) in enumerate(zip(good_new,good_old)):
        #     a,b = new.ravel()
        #     c,d = old.ravel()
        #     mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        #     frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        # img = cv2.add(frame,mask)

        # cv2.imshow('frame',img)
        # k = cv2.waitKey(30) & 0xff
        # if k == 27:
        #     break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    cap.release()

    return data


def lk_track_draw(fpath, fpath_dst, **kwargs):

    feature_params, lk_params = get_keywords_lktrack(**kwargs)
    cap = cv2.VideoCapture(fpath)
    k = cap.isOpened()
    if k == False:
        cap.open(fpath)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fpath_ex = os.path.splitext(fpath_dst)[-1]
    if fpath_ex == ".mp4" or fpath_ex == ".MP4":
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    else:
        raise ValueError("file extention error")
    
    writer = cv2.VideoWriter(fpath_dst, fmt, fps, (width, height))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)


    # data = list()
    frame_id = 0
    while(1):
        ret, frame = cap.read()
        if ret == False:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # print(err)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        if len(good_new) == 0:
            continue
        
        flow = good_new - good_old
        # data.append(dict(frame=frame_id, flow=flow.tolist(), 
        #                     position=good_new.tolist()
        # ))


        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)),(int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(),-1)
        img = cv2.add(frame, mask)
        writer.write(img)

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    writer.release()
