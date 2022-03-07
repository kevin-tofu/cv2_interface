import numpy as np
import cv2

"""
http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
https://qiita.com/icoxfog417/items/357e6e495b7a40da14d8
https://www.cresco.co.jp/blog/entry/16891/
https://www.programcreek.com/python/example/89313/cv2.calcOpticalFlowFarneback
https://stackoverflow.com/questions/41661517/drawing-results-of-calcopticalflowfarneback-in-opencv-without-loop

"""


def get_keywords_opticalflow(**kwargs):

        pyr_scale = kwargs['_pyr_scale']
        levels = kwargs['_levels']
        winsize = kwargs['_winsize']
        iterations = kwargs['_iterations']
        poly_n = kwargs['_poly_n']
        poly_sigma = kwargs['_poly_sigma']
        flags = kwargs['_flags']

        return pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags


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
    flow_np = np.asarray(flow).tolist()

    return flow_np



def opticalflow_dense_image_draw_base(img_prev,\
                                      img_next,\
                                    #   export='org+flow',\
                                      **kwargs):
    """
    """

    export = kwargs['export']
    img_prev_gray = cv2.cvtColor(img_prev, cv2.COLOR_BGR2GRAY)
    img_next_gray = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
    flow = opticalflow_dense_image_base(img_prev_gray,\
                                        img_next_gray,\
                                        **kwargs)
    hsv = np.zeros_like(img_next)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[...,0] = ang*180/np.pi/2
    

    if export == 'org-flow':
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img_ret = cv2.vconcat([rgb, img_next_gray])

    elif export == 'org+flow':
        img_ret = draw_flow(img_next_gray, flow, step=32)

    elif export == 'flow':
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
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



def opticalflow_dense_sum_video(fname, **kwargs):
        """
        """
        import math

        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags = get_keywords_opticalflow(**kwargs)

        cap = cv2.VideoCapture(fname)
        # pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags = 0.5, 3, 15, 3, 5, 1.2, 0

        k = cap.isOpened()
        if k == False:
            cap.open(fname)


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
            flow_mag = np.asarray(mag)
            flow_mag_sum = float(np.sum(flow_mag))
            # print(flow_mag_sum, type(flow_mag_sum))

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


def opticalflow_dense_draw(fname: str, \
                           fname_ex: str, \
                           export: str, \
                           **kwargs):
        """
        """
        
        pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags = get_keywords_opticalflow(**kwargs)
        # export = kwargs['export']

        cap = cv2.VideoCapture(fname)
        k = cap.isOpened()
        if k == False:
            cap.open(fname)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        if export == 'org-flow':
            writer = cv2.VideoWriter(fname_ex, fmt, fps, (width, 2*height))
        elif export == 'org+flow':
            writer = cv2.VideoWriter(fname_ex, fmt, fps, (width, height))
        elif export == 'flow':
            writer = cv2.VideoWriter(fname_ex, fmt, fps, (width, height))


        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        # flow_list = list()
        while(1):
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
                rgb2 = draw_flow(frame2, flow, step=32)
                writer.write(rgb2)

            elif export == 'flow':
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
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

        


def lk_track(fpath, feature_params, lk_params):

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
        
        flow = good_new - good_old
        data.append(dict(frame=frame_id, flow=flow.tolist(), 
                            position=good_new.tolist()
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


def lk_track_draw(fpath, fpath_ex, feature_params, lk_params):

    cap = cv2.VideoCapture(fpath)
    k = cap.isOpened()
    if k == False:
        cap.open(fpath)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    
    writer = cv2.VideoWriter(fpath_ex, fmt, fps, (width, height))

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
