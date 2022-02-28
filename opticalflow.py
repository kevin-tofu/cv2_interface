import numpy as np
import cv2


def opt_flow(fname, \
             pyr_scale, \
             levels, \
             winsize, \
             iterations, \
             poly_n, \
             poly_sigma, \
             flags):
        """
        """
        

        cap = cv2.VideoCapture(fname)
        # pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags = 0.5, 3, 15, 3, 5, 1.2, 0

        k = cap.isOpened()
        if k == False:
            cap.open(fname)


        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255

        flow_list = list()
        while(1):
            ret, frame2 = cap.read()
            if ret == False:
                break

            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
            flow_list.append(flow.tolist())

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


def opt_flow_draw(fname, \
                  fname_ex, \
                  pyr_scale, \
                  levels, \
                  winsize, \
                  iterations, \
                  poly_n, \
                  poly_sigma, \
                  flags):
        """
        """
        

        cap = cv2.VideoCapture(fname)
        k = cap.isOpened()
        if k == False:
            cap.open(fname)

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
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

            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
            # flow_list.append(flow.tolist())

            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            writer.write(rgb)

            # cv2.imshow('frame2',rgb)
            # k = cv2.waitKey(30) & 0xff
            # if k == 27:
            #     break
            # elif k == ord('s'):
            #     cv2.imwrite('opticalfb.png',frame2)
            #     cv2.imwrite('opticalhsv.png',rgb)
            prvs = next

        cap.release()
        writer.release()
        


def lk_track(fpath, feature_params, lk_params):

    cap = cv2.VideoCapture(fpath)

    # params for ShiTomasi corner detection
    # feature_params = dict( maxCorners = 100,
    #                     qualityLevel = 0.3,
    #                     minDistance = 7,
    #                     blockSize = 7 )

    # # Parameters for lucas kanade optical flow
    # lk_params = dict( winSize  = (15,15),
    #                 maxLevel = 2,
    #                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    # color = np.random.randint(0,255,(100,3))

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
    
    # params for ShiTomasi corner detection
    # feature_params = dict( maxCorners = 100,
    #                     qualityLevel = 0.3,
    #                     minDistance = 7,
    #                     blockSize = 7 )

    # # Parameters for lucas kanade optical flow
    # lk_params = dict( winSize  = (15,15),
    #                 maxLevel = 2,
    #                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)


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
        for i, (new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame, mask)
        writer.write(img)

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    writer.release()
