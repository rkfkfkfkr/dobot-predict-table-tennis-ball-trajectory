import cv2
import numpy as np
import imutils
import threading
import math
import timeit
import DobotDllType as dType
from numpy.linalg import inv

import cv2.aruco as aruco
import os

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

api = dType.load()

def dobot_get_start():
    
    dType.SetQueuedCmdClear(api)

    dType.SetHOMEParams(api, 200, 0, 20, 30, isQueued = 1) # x, y, z, r 
    dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued = 1) # velocity[4], acceleration[4]
    dType.SetPTPCommonParams(api, 250, 250, isQueued = 1) # velocityRatio(속도율), accelerationRation(가속율)
   
    dType.SetHOMECmd(api, temp = 0, isQueued = 1)

    dType.SetQueuedCmdStartExec(api)

def dobot_connect():

    CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound", 
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

    state = dType.ConnectDobot(api, "", 115200)[0]
    print("Connect status:",CON_STR[state])

    if (state == dType.DobotConnect.DobotConnect_NoError):
        dobot_get_start()

def segmentaition(frame):

    img_ycrcb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)
    y,cr,cb = cv2.split(img_ycrcb)

    _, cb_th = cv2.threshold(cb, 90, 255, cv2.THRESH_BINARY_INV)
    cb_th = cv2.erode(cb_th, None, iterations=2)
    cb_th = cv2.dilate(cb_th, None, iterations=2)

    return cb_th

def get_distance(x, y, imagePoints):
    
    objectPoints = np.array([[25,-5,0],
                            [35,-5,0],
                            [35,5,0],
                            [25,5,0],],dtype = 'float32')


    fx = float(470.5961)
    fy = float(418.18176)
    cx = float(275.7626)
    cy = float(240.41246)
    k1 = float(0.06950)
    k2 = float(-0.07445)
    p1 = float(-0.01089)
    p2 = float(-0.01516)

    cameraMatrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype = 'float64')
    distCoeffs = np.array([k1,k2,p1,p2],dtype = 'float64')
    _,rvec,t = cv2.solvePnP(objectPoints,imagePoints,cameraMatrix,distCoeffs)
    R,_ = cv2.Rodrigues(rvec)
            
    u = (x - cx) / fx
    v = (y - cy) / fy
    Qc = np.array([[u],[v],[1]])
    Cc = np.zeros((3,1))
    Rt = np.transpose(R)
    Qw = Rt.dot((Qc-t))
    Cw = Rt.dot((Cc-t))
    V = Qw - Cw
    k = -Cw[-1,0]/V[-1,0]
    Pw = Cw + k*V
    
    px = Pw[0]
    py = Pw[1]

    #print("px: %f, py: %f" %(px,py))

    return px,py

def find_ball(frame,cb_th,box_points):

    cnts = cv2.findContours(cb_th, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    px = None
    py = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
            px,py = get_distance(center[0], center[1],box_points)
            
            text = " %f , %f" %(px,py)
            cv2.putText(frame,text,center,cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

            print("px: %f, py: %f" %(px,py))

    return px,py

def findArucoMarkers(img, markerSize = 6, totalMarkers=250, draw=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray, arucoDict, parameters = arucoParam)

    if draw:
        cv2.aruco.drawDetectedMarkers(img,bboxs)
        #print(len(bboxs))

    if len(bboxs) > 0:
        return bboxs[0][0]
    else:
        return [0,0]

def move_dobot(of_x,of_y):

    offset_x = float(of_x * 10 + 89)#float((of_x + 6) * 10 - 5)
    offset_y = float(-1 * of_y * 10)#float(of_y * 10 - 3)

    offset_x = round(offset_x,2)
    offset_y = round(offset_y,2)
    last_index = 0
    
    length = math.sqrt(math.pow(offset_x,2) + math.pow(offset_y,2))

    if length > 50 and length < 270 and of_x > 0:

        print("offset_x: %f, offset_y: %f, length: %f \n" %(offset_x,offset_y,length))
        last_index = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, offset_x, offset_y, -10, 0, isQueued = 1)[0]

        dType.SetQueuedCmdStartExec(api)
        while last_index > dType.GetQueuedCmdCurrentIndex(api)[0]:
            dType.dSleep(100)

        dType.SetQueuedCmdStopExec(api)
    
def kalman_filter(z_meas, x_esti, P):
    """칼만필터 알고리즘 (매개변수 : 측정값, 추정값, 오차공분산)"""

    dt = 1
    A = np.array([[ 1, dt,  0,  0],
                  [ 0,  1,  0,  0],
                  [ 0,  0,  1, dt],
                  [ 0,  0,  0,  1]])
    H = np.array([[ 1,  0,  0,  0],
                  [ 0,  0,  1,  0]])
    Q = 1.0 * np.eye(4)
    R = np.array([[50,  0],
                  [ 0, 50]])

    # (1) Prediction.
    x_pred = A @ x_esti
    P_pred = A @ P @ A.T + Q

    # (2) Kalman Gain.
    K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)

    # (3) Estimation.
    x_esti = x_pred + K @ (z_meas - H @ x_pred)

    # (4) Error Covariance.
    P = P_pred - K @ H @ P_pred

    return x_esti, P

def get_pos(i,ball_x,ball_y):

    xc = float(ball_x[i])
    yc = float(ball_y[i])

    v = np.random.normal(0, 3)  # v: 위치의 측정잡음

    xpos_meas = xc  # x_pos_meas: 위치x의 측정값 (observable). 
    ypos_meas = yc  # y_pos_meas: 위치y의 측정값 (observable). 

    return np.array([xpos_meas, ypos_meas])


def predict_xy(ball_x,ball_y,p_x,p_y,last_x_esti, last_P,count):

    x_esti= None
    P = None

    # Initialization for estimation.
    x_0 = np.array([0,0,120,0])  # (x-pos, x-vel, y-pos, y-vel) by definition in book. / 추정값 초기위치
    P_0 = 100 * np.eye(4)


    if len(last_x_esti) == 0:
        for i in range(len(ball_x)):

            z_meas = get_pos(i,ball_x,ball_y)

            if i == 0:
                x_esti, P = x_0, P_0
        
            else:
                x_esti, P = kalman_filter(z_meas, x_esti, P)

            p_x.append(x_esti[0])
            p_y.append(x_esti[2])

            last_x_esti = x_esti
            last_P = P
    else:
        a_x = ball_x[count+1:len(ball_x)]
        a_y = ball_y[count+1:len(ball_x)]

        for i in range(len(a_x)):

            z_meas = get_pos(i,a_x,a_y)

            if i == 0:
                x_esti,P = last_x_esti,last_P

            else:
                x_esti, P = kalman_filter(z_meas, x_esti, P)

            p_x.append(x_esti[0])
            p_y.append(x_esti[2])

            last_x_esti = x_esti
            last_P = P

        #print(x_esti)

    return p_x,p_y,last_x_esti,last_P

def main():

    dobot_connect()

    cap = cv2.VideoCapture(0)

    ball_x = []
    ball_y = []
    x_vel = []
    y_vel = []

    plot_x = []
    plot_y = []
    p_x = []
    p_y = []
    
    count = 0

    last_x_esti = []
    last_P = None

    _,frame = cap.read()
    box_points = findArucoMarkers(frame)

    if len(box_points) > 2:
        print("Find Maker")

        start_t = timeit.default_timer()
        
        while(1):

            _,frame = cap.read()
            cb_th = segmentaition(frame)
            px,py = find_ball(frame,cb_th,box_points)

            if px != None and py != None:

                ball_x.append(px)
                ball_y.append(py)

                if float(px - 5) < 3 and float(px - 5) > 2 and len(plot_x) > 0:

                    of_x = plot_x[-1]
                    of_y = 5

                    move_dobot(of_x,of_y)
                    
                if py < ball_y[count] - 10:
                    
                    p_x,p_y,last_x_esti,last_P = predict_xy(ball_x, ball_y, p_x, p_y,last_x_esti, last_P,count)

                    plot_x.append(p_x[-1])
                    plot_y.append(p_y[-1])

                    print("p_x: %f, p_y: %f" %(plot_x[-1], plot_y[-1]))

                    of_x = plot_x[-1]
                    of_y = 5


                    if float(plot_y[-1]) < 45:
                        move_dobot(of_x,of_y)

                    count = len(ball_x) - 1

            elif px == None:
                if len(ball_x) > 1:
                    terminate_t = timeit.default_timer()
                    time = terminate_t - start_t
                    V = math.sqrt(math.pow(ball_x[0] - ball_x[-1],2) + math.pow(ball_y[0] - ball_y[-1],2))/time
                    print("average_V: %f cm/s" %(V))

                    start_t = timeit.default_timer()

                #if len(plot_x) > 0:

                    #plt.plot(ball_y, ball_x, 'b.')
                    #plt.plot(plot_y,plot_x,'r*')
                    #plt.xlabel('Y-pos. [cm]')
                    #plt.ylabel('X-pos. [cm]')

                    #terminate_t = timeit.default_timer()
                    #time = terminate_t - start_t
                    #V = math.sqrt(math.pow(ball_x[-1] - ball_x[0],2) + math.pow(ball_y[-1] - ball_y[0],2))/time

                    #print("average_V: %f cm/s" %(V))

                    #break
                ball_x.clear()
                ball_y.clear()
                count = 0
                last_x_esti = []
                last_p = None
            
            cv2.imshow('cam',frame)

            if cv2.waitKey(1) == 27:
                break

        plt.show()

        #dType.SetQueuedCmdStopExec(api)

        dType.DisconnectDobot(api)
        cap.release()
        cv2.destroyAllWindows()

main()
































    
