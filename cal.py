# 输出画框大小
from scipy.ndimage import standard_deviation

frame_out_w = 320 * 2
frame_out_h = 240
# camera (input) configuration 相机采集画框大小
frame_in_w = 320
frame_in_h = 240
IMAGE_ROW = 1  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 2  # 图片间隔，也就是合并成一张图后，一共有几列
import dlib
predictor_path = 'C:\\Users\\31583\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\dlib-19.20.0\\shape_predictor_68_face_landmarks.dat'
# 识别面部0
global predictor
detector = dlib.get_frontal_face_detector()
# 关键点检测器
predictor = dlib.shape_predictor(predictor_path)


import PIL.Image as Image
import cv2 as cv
import numpy as np
import time
import keyboard
import math
import pywt
from sklearn.decomposition import PCA
from scipy import signal
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

import multiprocessing
from multiprocessing import Manager
from sklearn.decomposition import FastICA

# 初始化
U = []
F = []
cam = 1
N = 500
face_cascade = cv.CascadeClassifier('C:/Users/31583/Desktop/python/data/haarcascade_frontalface_default.xml')
face_n = [0, 0, 0, 0]
face_n1 = [0, 0, 0, 0]
# 全局变量
cnt_1 = 0

# 画矩形
x = 200
y = 200
w = 20
h = 20


# 帧数判断：
# 输入：上一帧帧数cnt_n
# 输出：这一帧帧数cnt
def cnt_bult(cnt_n):
    cnt = cnt_n + 1
    return cnt


# 高斯滤波,输入：图像，高斯滤波参数，输出：高斯滤波后的图像
# 函数本身不产生画面
def gaussianbler(frame, x, level):
    y = x
    frame_gauss = cv.GaussianBlur(frame, (x, y), level)
    return frame_gauss


# 画出框框，输入：图片，框框尺寸  输出:在图像上画好的框
# 函数本身不产生画面
def rect_draw(frame, size, colour, cx):
    [x, y, w, h] = size
    frame_rect = cv.rectangle(frame, (x, y), (x + w, y + h), colour, cx)
    # cv.imshow("画框框",frame)
    return frame_rect


# 得到框内图像，输入：图片，框框尺寸 输出：框内独立的图像
def rect_get(frame, size):
    [x, y, w, h] = size
    rect = frame[y:(y + h), x:(x + w)]
    # cv.imshow("框框", rect)
    return rect


# 用函数求平均
def everage2(frame):
    eve = np.mean(frame)
    return eve


# 前后求插值
# 输入:这个时间的数值,前一帧数值，和当前帧数
# 输出：前一帧值，差值
def difference(v_n, v, i):
    div = list(map(lambda x: x[0] - x[1], zip(v, v_n)))
    v_n = v
    return v_n, div


def difference1(v_n, v, i):
    div = v - v_n[i]
    v_n[i] = v
    return v_n, div


# 直接fft再滤波#精简
# 滤波后fft,并输出图像
# 输入：目前的数组，如果数组长度没达到N就跳过此函数，如果达到N就进行fft
# 输入：fs采样频率（在外部计算完成后输入
# 输出：fft后的数组
def fft_bult2(U, fs, N, i):
    from scipy.fftpack import fft, ifft
    # 计算频率下限对应的fft下标(50/min)
    f_low = int(60 / ((fs / N) * 60))
    # 计算频率上限对应的fft下标(160/min)
    f_high = int(30 * fs / ((fs / N) * 60))
    # print('f_low',f_low)
    # print('f_high',f_high)
    # print('high puls is ',30*fs)
    # print('per puls is ',(fs/N)*60)
    Uf_lv = np.zeros((f_high), dtype=np.float)
    if (len(U) >= N):
        # print("start to fft")
        # 直接进行fft
        Uf = np.abs(fft(U))  # 取绝对值
        Uf1 = Uf[range(int(N / 2))]  # 由于对称性，只取一半区间
        # UF1[0:f_low] = 0
        # 滤波后的fft
    return Uf1


def fft_bult3(U, fs, N, i):
    from scipy.fftpack import fft, ifft
    # 计算频率下限对应的fft下标(50/min)
    f_low = int(10 / ((fs / N) * 60))
    # 计算频率上限对应的fft下标(160/min)
    f_high = int(30 / ((fs / N) * 60))
    # print('f_low',f_low)
    # print('f_high',f_high)
    # print('high puls is ',30*fs)
    # print('per puls is ',(fs/N)*60)
    Uf_lv = np.zeros((f_high), dtype=np.float)
    if (len(U) >= N):
        # print("start to fft")
        # 直接进行fft
        Uf = np.abs(fft(U))  # 取绝对值
        Uf1 = Uf[range(int(N / 2))]  # 由于对称性，只取一半区间
        # 滤波后的fft
    return Uf1


# 产生数组#以便之后做fft
# 有除去突变点
# 根据外界条件产生数组，由于不是每一帧画面都要存入数组
# 输入：这一帧对应的值v,到目前已经生成的数组U，有效帧数cnt_1
# 输出：当前帧数下产生的数组
def arry_bult(v):
    global F
    global M
    global cnt_arry
    global L
    cnt_arry = cnt_arry + 1  # 数组中加入了第cnt_arry个数
    b = 0
    if ((cnt_arry >= 5) & (cnt_arry <= L)):  # 5-62个数求平均
        a = abs(v)
        F.append(a)
    M = np.mean(F)  # 得到平均像素值
    if (cnt_arry > L):  # 前62个数舍弃
        if (abs(v) <= 5 * M):  # 当前数如果小于3倍的5-62个数的平均值，则认为是好点
            b = v
        else:
            b = 0
    return b


# 位移数组
def arry_move(v):
    global M
    global cnt_arry
    cnt_arry = cnt_arry + 1
    if (abs(v) <= 5 * M):
        b = v
    else:
        b = 0
    return b


# 求心率
# 求最大值频率 @在U长度达到N时经行
# 返回最大频率
def max_p(U, fs, N, i):
    # 计算频率下限对应的fft下标(50/min)
    f_low = int(50 / ((fs / N) * 60))
    # 计算频率上限对应的fft下标(200/min)
    f_high = int(145 / ((fs / N) * 60))
    M = U[f_low:f_high].argmax()
    F_max = (M + f_low) * fs / N * 60
    print("~~~~~~~~~processed xin: ", i, "：", F_max, '~~~~~~~~')
    return F_max


# 求心率2
# 求最大值频率 @在U长度达到N时经行
# 返回最大频率
def max_p2(U, fs, N, i):
    # print("max begin")
    # 归一化
    U = noramlization(U)
    f_100 = int(100 / ((fs / N) * 60))
    # 计算频率下限对应的fft下标(50/min)
    f_low = int(60 / ((fs / N) * 60))
    #U[:(f_low+1)] = 0
    # 计算频率上限对应的fft下标(200/min)
    f_high = int(145 / ((fs / N) * 60))
    M1 = U[(f_low):f_high].argmax()
    f1 = U[M1 + f_low]
    lv1 = ((M1 + f_low) * fs / N * 60)
    # print("panduan kaishi")
    if (lv1 >= 100):
        # print("if kaishi")
        M2 = U[f_low:f_100].argmax()
        f2 = U[M2 + f_low]
        lv2 = ((M2 + f_low) * fs / N * 60)
        # print("if end")
        if (((abs(lv2 - lv1 / 2) <= 13) & (abs(f1 - f2) <= 0.3)) or ((abs(f1 - f2) <= 0.1) & (abs(M1 - M2) > 20))):
            F_max = lv2
            # print("if 2 kaishi")
            print('lv1~~~~~~~~~~~~~~~~~~~~~~', lv1)
            print('lv2~~~~~~~~~~~~~~~~~~~~~~', lv2)
            # print('div~~~~~~~~~~~~~~~~~~~~~~', lv1 / 2 - lv2)
            # print('div zhi~~~~~~~~~~~~~~~~~~', f1 - f2)
        else:
            # print("else 2 kaishi")
            F_max = lv1
            """print('lv1~~~~~~~~~~~~~~~~~~~~~~', lv1)
            print('lv2~~~~~~~~~~~~~~~~~~~~~~', lv2)
            print('div~~~~~~~~~~~~~~~~~~~~~~', lv1 / 2 - lv2)
            print('div zhi~~~~~~~~~~~~~~~~~~', f1 - f2)"""
        print("if else end")
    else:
        F_max = lv1
        # print("else end")

    print("~~~~~~~~~processed xin: ", i, "：", F_max, '~~~~~~~~')
    # print("max end")

    return F_max


def max_p3(U, fs, N, i):
    # print("max begin")
    # 归一化
    U = noramlization(U)
    f_100 = int(100 / ((fs / N) * 60))
    # 计算频率下限对应的fft下标(50/min)
    f_low = int(10/ ((fs / N) * 60))
    # 计算频率上限对应的fft下标(200/min)
    f_high = int(30 / ((fs / N) * 60))
    M1 = U[f_low:f_high].argmax()
    f1 = U[M1 + f_low]
    lv1 = ((M1 + f_low) * fs / N * 60)
    # print("panduan kaishi")
    if (lv1 >= 100):
        # print("if kaishi")
        M2 = U[f_low:f_100].argmax()
        f2 = U[M2 + f_low]
        lv2 = ((M2 + f_low) * fs / N * 60)
        # print("if end")
        if (((abs(lv2 - lv1 / 2) <= 13) & (abs(f1 - f2) <= 0.3)) or ((abs(f1 - f2) <= 0.1) & (abs(M1 - M2) > 20))):
            F_max = lv2
            # print("if 2 kaishi")
            print('lv1~~~~~~~~~~~~~~~~~~~~~~', lv1)
            print('lv2~~~~~~~~~~~~~~~~~~~~~~', lv2)
            # print('div~~~~~~~~~~~~~~~~~~~~~~', lv1 / 2 - lv2)
            # print('div zhi~~~~~~~~~~~~~~~~~~', f1 - f2)
        else:
            # print("else 2 kaishi")
            F_max = lv1
            """print('lv1~~~~~~~~~~~~~~~~~~~~~~', lv1)
            print('lv2~~~~~~~~~~~~~~~~~~~~~~', lv2)
            print('div~~~~~~~~~~~~~~~~~~~~~~', lv1 / 2 - lv2)
            print('div zhi~~~~~~~~~~~~~~~~~~', f1 - f2)"""
        print("if else end")
    else:
        F_max = lv1
        # print("else end")

    print("~~~~~~~~~processed xin: ", i, "：", F_max, '~~~~~~~~')
    # print("max end")

    return F_max


# 画出图像
def draw_ax(y, X):
    import matplotlib.pyplot as plt
    # y纵坐标
    plt.subplot(311)
    plt.plot(y, 'r')
    plt.plot(X)
    plt.title('PULS')

    plt.subplot(312)
    plt.plot(y, 'r')
    plt.title('PULS')

    plt.subplot(313)
    plt.plot(X)
    plt.title('PULS1')

    plt.show()
    print("y",y)
    print("X",X)

def abc(x):
    global n
    global l
    if ((x.event_type == "down") and (x.name == "1")):
        n += 1
        l += 1
    if ((x.event_type == "down") and (x.name == "2")):
        n -= 1
        l -= 1
    global n2
    global l2
    if ((x.event_type == "down") and (x.name == "3")):
        n2 += 1
        l2 += 1
    if ((x.event_type == "down") and (x.name == "4")):
        n2 -= 1
        l2 -= 1


n = 15
l = 15
n2 = 15
l2 = 15
u = 1

##############################################
# 1.获得稳定的人脸和ROI区域
def get_stable_face(frame, flag,roi_cnt,roi,roi_cent_U1,roi_U1,roi_cent_U2,roi_U2):
    global face_cascade
    global face_n1
    global i_face
    global cnt_face  # 用于获得失去人脸的帧数
    # 判断是否使用meanshift
    judge = 0
    close = 0
    global n
    global l
    global u
    if (u == 1):
        keyboard.hook(abc)
        u = 0
    print("```````````````````````````````````n", n)
    # 识别面部0
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.4, 3)

    # 挑选出最宽的脸
    w_max = np.int32(0)
    face_max = np.array([0, 0, 0, 0], dtype='int32')
    for (x, y, w, h) in faces:  # 检测到三张人脸，从三张人脸中选出最宽的那一张
        if (w > w_max):
            w_max = w
            face_max = [x, y, w, h]  # 实时监测的人脸

    if (flag == False):
        cnt_face = 0
        # 判断是否持续识别一张脸
        if (face_max[0] == 0):
            i_face = 0
            flag = False
            face_n1 = face_max

        elif ((abs(face_max[0] - face_n1[0])) >= 16):
            i_face = 0
            flag = False
            face_n1 = face_max

        elif (i_face <= 5):  # 稳定时长
            flag = False
            i_face = i_face + 1  # 每循环一次加一
            face_n1 = face_max

        elif (i_face == 6):  # 当i_face计时6次时，跳出循环
            flag = True  # 令flag = true，跳出此if循环
            i_face = i_face
            face_n1 = face_max

        # 划定roi：
        # ROI[0]
        [x, y, w, h] = face_max
        w_w = w * 0.35
        h_h = h * 0.225
        e_x = w * 0.325
        e_y = h * 0.025
        roi_cnt = np.array([e_x, e_y, w_w, h_h], dtype='int32')
        roi = np.array([e_x + x, e_y + y, w_w, h_h], dtype='int32')
        # ROI[1]
        w_w = w * 0.125
        h_h = h * 0.35
        e_x = w * 0.4375
        e_y = h * 0.25
        roi_cent_U1 = np.array([e_x, e_y, w_w, h_h], dtype='int32')
        roi_U1 = np.array([e_x + x, e_y + y, w_w, h_h], dtype='int32')
        # ROI[2]
        w_w = w * 0.16
        h_h = h * 0.15
        e_x = w * 0.64
        e_y = h * 0.50
        roi_cent_U2 = np.array([e_x, e_y, w_w, h_h], dtype='int32')
        roi_U2 = np.array([e_x + x, e_y + y, w_w, h_h], dtype='int32')
        return face_max,roi_cnt,flag,roi,roi_cent_U1,roi_U1,roi_cent_U2,roi_U2


# 2.根据传入face,使用meanshift计算得出预测位置
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade.load(r'C:/Users/31583/Desktop/python/data/haarcascade_frontalface_default.xml')


def meanshift(face_live, frame):
    c, r, w, h = face_live
    track_window = (c, r, w, h)

    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 0., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    # 设置终止条件
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    # 调用meanshift获取新的位置
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    # 画出它的位置
    return track_window


# 3.阈值判断确定使用meanshift还是使用稳定人脸
def choose_face(flag,face_max, track_window):
    x1, y1, w1, h1 = face_max
    x2, y2, w2, h2 = track_window
    if (abs(x1 - x2) + abs(y1 - y2) >= 15):
        face_max = track_window
        #("1111111111111111111111",track_window)
        flag = 1
    return flag,face_max

########################################################
def detect_one_face(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 3)
    if len(faces) == 0:
        return (0, 0, 0, 0)
    return faces[0]

def kalman_filter_tracker(frame,face_live):
    c, r, w, h = face_live
    temp1 = w
    temp2 = h
    #print('helloworld')
    #print(w)
    #print(h)
    kalman = cv2.KalmanFilter(4, 2, 0)

    state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')  # initial position
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                        [0., 1., 0., .1],
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state
    # use prediction or posterior as your tracking result

    img_width = frame.shape[0]
    img_height = frame.shape[1]

    prediction = kalman.predict()

    pos = 0
    c, r, w, h = detect_one_face(frame)
    if w != 0 and h != 0:
        state = np.array([c + w / 2, r + h / 2, 0, 0], dtype='float64')
        # kalman.statePost = state
        measurement = (np.dot(kalman.measurementNoiseCov, np.random.randn(2, 1))).reshape(-1)
        measurement = np.dot(kalman.measurementMatrix, state) + measurement
        posterior = kalman.correct(measurement)
        pos = (posterior[0], posterior[1])
    else:
        measurement = (np.dot(kalman.measurementNoiseCov, np.random.randn(2, 1))).reshape(-1)
        measurement = np.dot(kalman.measurementMatrix, state) + measurement
        pos = (prediction[0], prediction[1])

    # 第一行为观测值，第二行为预测值
    process_noise = np.sqrt(kalman.processNoiseCov[0, 0]) * np.random.randn(4, 1)
    state = np.dot(kalman.transitionMatrix, state) + process_noise.reshape(-1)
    x,y,w,h = state
    track_window = (abs(int(x - temp1 / 2)), abs(int(y - temp2 / 2)), temp1, temp2)
    """print('='*20)
    print(state)
    print(x)
    print(y)
    print(w)
    print(h)
    print('='*20)"""
    return track_window
########################################################

# 人脸识别2
# 输入：画面，当前的flag,当前识别出来的roi
# 输出：实时显示的人脸范围，输出人脸roi区域，flag
def face_detect2(frame, flag, roi_cent, roi, track_window, MSflag):
    global face_cascade
    global face_n1
    global i_face
    global cnt_face  # 用于获得失去人脸的帧数
    # 判断是否使用meanshift
    judge = 0
    close = 0
    global n
    global l
    global u
    if (u == 1):
        keyboard.hook(abc)
        u = 0
    #print("```````````````````````````````````n", n)
    # 识别面部0
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.4, 3)

    # 挑选出最宽的脸
    w_max = np.int32(0)
    face_max = np.array([0, 0, 0, 0], dtype='int32')
    for (x, y, w, h) in faces:  # 检测到三张人脸，从三张人脸中选出最宽的那一张
        if (w > w_max):
            w_max = w
            face_max = [x, y, w, h]  # 实时监测的人脸

    if (flag == False):
        cnt_face = 0
        # 判断是否持续识别一张脸########################################
        if (face_max[0] == 0):
            close = 2
            i_face = 0
            flag = False
            face_n1 = face_max

        elif ((abs(face_max[0] - face_n1[0])) >= 10):
            close = 0
            i_face = 0
            flag = False
            face_n1 = face_max
        elif (i_face <= 5):  # 稳定时长
            close = 0
            flag = False
            i_face = i_face + 1  # 每循环一次加一
            face_n1 = face_max
        elif (i_face == 6):  # 当i_face计时6次时，跳出循环
            close = 0
            flag = True  # 令flag = true，跳出此if循环
            i_face = i_face
            face_n1 = face_max

        # 划定roi：
        [x, y, w, h] = face_max
        # roi大小
        # w_w = w * 0.4
        # h_h = h * 0.9
        w_w = w * 0.2
        h_h = h * 0.2

        # w_w = 8
        # h_h = 8
        # roi起点坐标
        e_x = w * 0.3
        e_y = h * 0.05
        roi_cent = np.array([e_x, e_y, w_w, h_h], dtype='int32')
        roi = np.array([e_x + x, e_y + y, w_w, h_h], dtype='int32')

        # roi_cent就是roi区域

    # 当识别的人脸已经稳定后
    if (flag == True):
        if ((abs(face_max[0] - face_n1[0])) <= 5):  # 如果人脸没有偏离，即实时检测的人脸和固定的人脸位置差别不大
            face_max = face_n1  # 将当前的识别到的人脸给face_max
            if (MSflag == 1):
                face_n1 = track_window
                face_max = track_window

            [x, y, w, h] = face_n1  # 获得当前人脸的起始坐标以及宽和高
            cnt_face = 0

            # 划定roi：
            # roi大小
            # w_w = w * 0.4
            # h_h = h * 0.9

            w_w = w * 0.2
            h_h = h * 0.2

            # w_w = 8
            # h_h = 8
            # roi起点坐标
            e_x = w * 0.3
            e_y = h * 0.05
            roi_cent = np.array([e_x, e_y, w_w, h_h], dtype='int32')
            roi = np.array([e_x + x, e_y + y, w_w, h_h], dtype='int32')
            judge = 1
            # roi_cent就是roi区域
        elif ((abs(face_max[0] - face_n1[0])) > 5 and (abs(face_max[0] - face_n1[0])) <= 30):
            face_max = face_n1  # 将当前的识别到的人脸给face_max

            # 给meanshift计算值
            if (MSflag == 1):
                face_n1 = track_window
                face_max = track_window

            [x, y, w, h] = face_n1  # 获得当前人脸的起始坐标以及宽和高
            cnt_face = 0

            # 划定roi：
            # roi大小
            # w_w = w * 0.4
            # h_h = h * 0.9

            w_w = w * 0.2
            h_h = h * 0.2
            # w_w = 8
            # h_h = 8
            # roi起点坐标
            e_x = w * 0.3
            e_y = h * 0.05
            roi_cent = np.array([e_x, e_y, w_w, h_h], dtype='int32')
            roi = np.array([e_x + x, e_y + y, w_w, h_h], dtype='int32')
            judge = 2

        elif ((abs(face_max[0] - face_n1[0])) > 30):  # 如果人脸偏移，即实时检测的人脸和固定的人脸位置差别过大
            #print('人脸偏离的帧数:', cnt_face)
            if (cnt_face >= 20):  # 如果偏离20帧 退出测量心率
                close = 3
            else:
                face_max = face_n1
                [x, y, w, h] = face_n1
                cnt_face = cnt_face + 1
                # 划定roi：
                # roi大小
                # w_w = w * 0.4
                # h_h = h * 0.9

                w_w = w * 0.2
                h_h = h * 0.2

                # w_w = 8
                # h_h = 8
                # roi起点坐标
                e_x = w * 0.3
                e_y = h * 0.05
                roi_cent = np.array([e_x, e_y, w_w, h_h], dtype='int32')
                roi = np.array([e_x + x, e_y + y, w_w, h_h], dtype='int32')
                # roi_cent就是roi区域
    #print("roi", roi_cent[2], roi_cent[3])
    return face_max, roi_cent, flag, close, roi, judge


# 人脸识别2
# 输入：画面，当前的flag,当前识别出来的roi
# 输出：实时显示的人脸范围，输出人脸roi区域，flag
def face_detect4(frame, flag, roi_cent, roi):
    global face_cascade
    global face_n1

    global i_face
    global cnt_face  # 用于获得失去人脸的帧数

    close = 0
    global n2
    global l2
    #print("``````````````````````````````````````n2", n2)
    # 识别面部0
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.4, 3)

    # 挑选出最宽的脸
    w_max = np.int32(0)
    face_max = np.array([0, 0, 0, 0], dtype='int32')
    for (x, y, w, h) in faces:  # 检测到三张人脸，从三张人脸中选出最宽的那一张
        if (w > w_max):
            w_max = w
            face_max = [x, y, w, h]  # 实时监测的人脸

    if (flag == False):
        cnt_face = 0
        # 判断是否持续识别一张脸########################################
        if (face_max[0] == 0):
            close = 2
            i_face = 0
            flag = False
            face_n1 = face_max

        elif ((abs(face_max[0] - face_n1[0])) >= 10):
            if (face_max[2] <= 150):  # 当画面尺寸太小，即未能收集到足够的像素点时
                close = 1  # 提醒plese be closer
                i_face = 0  # i_face被清零，重新开始稳定时长
                flag = False
                face_n1 = face_max
            else:
                close = 0
                i_face = 0
                flag = False
                face_n1 = face_max
        elif (i_face <= 5):  # 稳定时长
            if (face_max[2] <= 150):  # 当画面尺寸太小，即未能收集到足够的像素点时
                close = 1  # 提醒plese be closer
                flag = False
                i_face = 0  # i_face被清零，重新开始稳定时长
                face_n1 = face_max
            else:
                close = 0
                flag = False
                i_face = i_face + 1  # 每循环一次加一
                face_n1 = face_max
        elif (i_face == 6):  # 当i_face计时6次时，跳出循环
            close = 0
            flag = True  # 令flag = true，跳出此if循环
            i_face = i_face
            face_n1 = face_max

        # 划定roi：
        [x, y, w, h] = face_max
        # roi大小
        w_w = w * 0.4
        h_h = h * 0.9
        # roi起点坐标
        e_x = w * 0.3
        e_y = h * 0.05
        roi_cent = np.array([e_x, e_y, w_w, h_h], dtype='int32')
        roi = np.array([e_x + x, e_y + y, w_w, h_h], dtype='int32')
        # roi_cent就是roi区域

    # 当识别的人脸已经稳定后
    if (flag == True):
        if (face_max[0] != 0):  # 如果人脸没有跟丢，即实时可以检测到人脸
            if ((abs(face_max[0] - face_n1[0])) <= 30):  # 如果人脸没有偏离，即实时检测的人脸和固定的人脸位置差别不大
                face_max = face_n1  # 将当前的识别到的人脸给face_max
                [x, y, w, h] = face_n1  # 获得当前人脸的起始坐标以及宽和高
                cnt_face = 0

                # 划定roi：
                # roi大小
                w_w = w * 0.4
                h_h = h * 0.9
                # roi起点坐标
                e_x = w * 0.3
                e_y = h * 0.05
                roi_cent = np.array([e_x, e_y, w_w, h_h], dtype='int32')
                roi = np.array([e_x + x, e_y + y, w_w, h_h], dtype='int32')
                # roi_cent就是roi区域

                face_max = np.array([0, 0, 0, 0], dtype='int32')  # 清除外框
            else:  # 如果人脸偏移，即实时检测的人脸和固定的人脸位置差别过大
                #print('人脸偏离的帧数:', cnt_face)
                if (cnt_face >= 20):  # 如果偏离20帧 退出测量心率
                    close = 3
                else:
                    face_max = face_n1
                    [x, y, w, h] = face_n1
                    cnt_face = cnt_face + 1
                    # 划定roi：
                    # roi大小
                    w_w = w * 0.4
                    h_h = h * 0.9
                    # roi起点坐标
                    e_x = w * 0.3
                    e_y = h * 0.05
                    roi_cent = np.array([e_x, e_y, w_w, h_h], dtype='int32')
                    roi = np.array([e_x + x, e_y + y, w_w, h_h], dtype='int32')
                    # roi_cent就是roi区域
                    face_max = np.array([0, 0, 0, 0], dtype='int32')  # 清除外框

        else:  # 如果没有识别到人脸
            #print('跟丢人脸的帧数:', cnt_face)
            if (cnt_face >= 20):  # 跟丢人脸20帧，退出测量心率
                close = 3
            else:
                face_max = face_n1
                [x, y, w, h] = face_n1
                cnt_face = cnt_face + 1

                # 划定roi：
                # roi大小
                w_w = w * 0.4
                h_h = h * 0.9
                # roi起点坐标
                e_x = w * 0.3
                e_y = h * 0.05
                roi_cent = np.array([e_x, e_y, w_w, h_h], dtype='int32')
                roi = np.array([e_x + x, e_y + y, w_w, h_h], dtype='int32')
                # roi_cent就是roi区域
                face_max = np.array([0, 0, 0, 0], dtype='int32')  # 清除外框

    #print("roi", roi_cent[2], roi_cent[3])
    return roi_cent, roi


# 归一化
def noramlization(data):
    minVals = min(data)
    maxVals = max(data)
    ranges = maxVals - minVals
    y = (data - minVals) / ranges
    y = y - np.mean(y)
    return y


def calculate(U_face, U_face2, N1, puls, puls2, fs, flag1):
    while (True):
        try:
            if (flag1.value == 0):
                # print("`````````````````````````0",len(U_face) )
                if (len(U_face) >= 150):  # 当数组长度为100时开始计算心率
                    # print("calculate start")
                    # print('fs = ', fs.value)

                    fft_face = fft_bult2(U_face, fs.value, len(U_face), 1)  # 对信号进行傅里叶变换

                    #print("`````````````````````````fft end")
                    # 求最大值
                    # 求心率
                    puls.value = max_p2(fft_face, fs.value, len(U_face), 1)
                    #print("max end")
                    #print("")

                    fft_face2 = fft_bult2(U_face2, fs.value, len(U_face2), 1)  # 对信号进行傅里叶变换
                    #print("`````````````````````````fft2")
                    # 求最大值
                    # 求心率
                    puls2.value = max_p2(fft_face2, fs.value, len(U_face2), 1)
                    print("```````````````````````fs",fs.value)
                    #print("`````````````````````````U_face", U_face)
                    #print("`````````````````````````U_face2", U_face2)
                    #print("max2 end")
                    #print("")
                    flag1.value = 1

                else:
                    print("`````````````````````````1", len(U_face))
                    flag1.value = 1
        except:
            print("`````````````````````````2", len(U_face))
            flag1.value = 1
            continue
        else:
            flag1.value = 1
            continue


def picture_roi(frame, face_live, roi, roi_u1,roi_u2,roi2):
    # 画出实时脸的框
    frame_rect = frame.copy()
    frame_rect = rect_draw(frame_rect, face_live, (255, 0, 0), 2)
    # 画出roi的框
    frame_rect = rect_draw(frame_rect, roi, (255, 0, 0), 2)
    frame_rect = rect_draw(frame_rect, roi_u1, (255, 0, 0), 2)
    #frame_rect = rect_draw(frame_rect, roi_u2, (255, 0, 0), 2)
    #frame_rect = rect_draw(frame_rect, roi2, (255, 255, 255), 2)
    return frame_rect


def picture_time(frame_rect, cnt_arry, L, cnt_3, cnt_4, flag_roi,fs):
    # 画出进度条
    fs_s = int(fs) + 1
    frame_rect = rect_draw(frame_rect, (0, 0, int(frame_out_w * ((cnt_arry - L + 1) / 60)), 10), (0, 255, 0), 10)
    # 显示时间
    if ((cnt_arry % fs_s == 0) & (cnt_arry != 0) & (cnt_arry > (L - fs_s))):
        cnt_3 += 1
    if (cnt_arry > (L )):
        cv.putText(frame_rect, 'time is ' + str((cnt_3 - 16)) + ' s', (20, 170), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 250),
                   1)  # 显示时间
    # 显示准备
    if (flag_roi == True):
        if ((cnt_arry % fs_s == 0) & (cnt_arry != 0) & (cnt_arry <= (L - fs_s))):
            cnt_4 -= 1
        if (cnt_arry <= (L - fs_s/2)):
            if (cnt_4 != 0):
                cv.putText(frame_rect, 'ready ' + str((int(cnt_4))), (20, 170), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 250),
                           1)  # 显示准备
            else:
                cv.putText(frame_rect, 'go! ', (20, 170), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 250), 1)  # 显示准备
        if (cnt_arry == (L )):
            cv.putText(frame_rect, '                          ', (20, 170), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 250),
                       1)  # 显示准备
    else:
        cv.putText(frame_rect, '                          ', (20, 170), cv.FONT_HERSHEY_TRIPLEX, 1.0,
                   (0, 0, 250), 1)  # 显示准备
    return frame_rect, cnt_3, cnt_4


def picture_heart(frame_rect, puls_n, puls_n2):
    cv.putText(frame_rect, 'heart rate : ' + str(float(puls_n)), (20, 50), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 250),
               1)  # 显示 “ 心率值 ”
    cv.putText(frame_rect, 'heart rate : ' + str(float(puls_n2)), (20, 100), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 250),
               1)  # 显示 “ 心率值 ”
    return frame_rect


def guss_roi(rect_face,k):
    for i in range(k):
        rect_face = gaussianbler(rect_face, 51, 0)
    return rect_face

# ica
def fastica(n):
    fast_ica = FastICA(n_components=3)
    Sr = fast_ica.fit_transform(n)
    return Sr


def noramlization1(arr):
    a = []
    for x in arr:
        x = float(x - arr.mean()) / arr.std()
        a.append(x)
    return (a)


# *****************************************************************************************
#
#
#
# 得到对应的RGB三个序列
def get_RGB_vary(M0, M1, M2, cnt_arry, L, rect_face, rect_face_n0, rect_face_n1, rect_face_n2):
    cc1 = 0
    # 返回RGB对应的cn值
    cn0 = 0
    cn1 = 0
    cn2 = 0
    if (cnt_arry > L):
        for i in range(len(rect_face)):  # 对每一行数据依次循环
            Blue = everage2(rect_face[i, :, 0])  # 取第i行的蓝色通道的像素值的平均值
            Green = everage2(rect_face[i, :, 1])  # 取第i行绿色通道的像素值的平均值
            Red = everage2(rect_face[i, :, 2])  # 取第i行的红色通道的像素值的平均值
            cc1 += 1
            # 前后求差
            rect_face_n0, div_face_0 = difference1(rect_face_n0, Blue, cc1 - 1)
            rect_face_n0[cc1 - 1] = int(rect_face_n0[cc1 - 1])

            rect_face_n1, div_face_1 = difference1(rect_face_n1, Green, cc1 - 1)
            rect_face_n1[cc1 - 1] = int(rect_face_n1[cc1 - 1])

            rect_face_n2, div_face_2 = difference1(rect_face_n2, Red, cc1 - 1)
            rect_face_n2[cc1 - 1] = int(rect_face_n2[cc1 - 1])
            # 对相邻两帧画面各像素值的差值做二值化处理
            if (div_face_0 > 0):
                cn0 += 1
            if (div_face_1 > 0):
                cn1 += 1
            if (div_face_2 > 0):
                cn2 += 1
        M0.append(cn0)
        M1.append(cn1)
        M2.append(cn2)
    return M0, M1, M2


# ica处理
def ica(M0, M1, M2):
    s = np.array([M0, M1, M2])  # [3,x]
    ica = FastICA(n_components=3)
    s = s.T  # [x,3]
    u = ica.fit_transform(s)
    u = u.T
    return u  # [3,x]


import pandas as pd


# 返回与绿色相关性最强的下标，G指绿色序列
def dimension(u, G):
    H0 = u[0]
    H1 = u[1]
    H2 = u[2]
    BGR = np.array([H0, H1, H2, G])
    dfBGR = pd.DataFrame(BGR.T, columns=['a', 'b', 'c', 'G'])
    r1 = dfBGR.a.corr(dfBGR.G)
    r2 = dfBGR.b.corr(dfBGR.G)
    r3 = dfBGR.c.corr(dfBGR.G)
    res = [r1, r2, r3]
    max = 0
    k = 0
    for i in range(3):
        if (abs(res[i]) > max):
            max = abs(res[i])
            k = i
    return u[k]


#
#
#
# *****************************************************************************************
def get_vary_normal(cnt_arry,L,rect_face):
    # ···············中间变量初始化···········
    cn = 0
    cn2 = 0
    cn3 = 0
    if (cnt_arry > L):
        cn = everage2(rect_face[:, :, 0])  # 取第i行的1通道的像素值的平均值
        cn2 = everage2(rect_face[:, :, 1])  # 取第i行的1通道的像素值的平均值
        cn3 = everage2(rect_face[:, :, 2])
    return cn,cn2,cn3


def get_vary(changdu,zhi,picture,cnt_arry,L,rect_face,c):
    # ···············中间变量初始化···········
    if (cnt_arry > L):
        #print("```````````````````````````````len", len(rect_face))
        if(cnt_arry==L+1):
            for i in range(len(rect_face)):
                picture.append([])
            changdu = len(rect_face)

        if((changdu) > (len(rect_face))):
            changdu = len(rect_face)


        for i in range(changdu):
            picture[(i)].append(np.mean(rect_face[(i), :, c]))


    return picture,zhi,changdu


def get_vary_ch(changdu,picture,picture2,picture3,cnt_arry,L,rect_face):
    # ···············中间变量初始化···········
    if (cnt_arry > L):
        #print("```````````````````````````````len", len(rect_face))
        if(cnt_arry==L+1):
            for i in range(len(rect_face)):
                picture.append([])
                picture2.append([])
                picture3.append([])

            changdu = len(rect_face)

        if((changdu) > (len(rect_face))):
            changdu = len(rect_face)


        for i in range(changdu):
            picture[(i)].append(np.mean(rect_face[(i), :, 0]))

        for i in range(changdu):
            picture2[(i)].append(np.mean(rect_face[(i), :, 1]))

        for i in range(changdu):
            picture3[(i)].append(np.mean(rect_face[(i), :, 2]))


    return picture,picture2,picture3,changdu


# 计算心率,直方图方法
def cont_zhi_ch(cnt_frame,picture,picture2,picture3,changdu,fs):
    #print("``````````````````face", len(face))
    p = []
    if ( (cnt_frame>150)):
    # ································求出均值与方差··························
        if (len(picture[1])>210):
            for i in range(changdu):
                picture[i] = picture[i][30:210]
                picture2[i] = picture2[i][30:210]
                picture3[i] = picture3[i][30:210]


        #算鼻子部分
        print("```````````````````````````1")
        zhi = ([0] * (len(picture[1]) -1))
        a = ([0] * (len(picture[1]) -1)) # 多少个时刻，应该与picture中最短的时刻相匹配
        #print("```````````````````````````2")
        for i in range(changdu):  #循环每一行

            X = np.empty((len(picture[i]), 3))

            X[:, 0] = picture[i].copy()
            X[:, 1] = picture2[i].copy()
            X[:, 2] = picture3[i].copy()

            x = 3 * X[ : , 2] - 2 * X[: , 1]
            Y = 1.5 * X[: , 2] + X[: , 1] - 1.5 * X[:, 1]
            x = lvbo(x, 60, 95, fs)
            Y = lvbo(Y, 60, 95, fs)

            x_std = np.std(x, ddof=1)
            Y_std = np.std(Y, ddof=1)
            C = x - (x_std / Y_std) * Y
            C = np.array(C)

            data = lvbo(C, 55, 150,fs)


            diff_x1 = np.diff(data)
            cn = 0
            for j in range(len(diff_x1)):  #第i行的第j个时刻
                if diff_x1[j] > 0:             #最后一行在上一个时刻没有出现，所以其j少一个，即其不为a[Tmax]贡献
                    a[j]+=1

                else:
                    for k in range(20):
                        if diff_x1[j - k] > 0:
                            cn += 1
                        if cn > 6:
                            a[j] += 1
                        else:
                            a[j] += 0
                        cn = 0


        #print("```````````````````````````3")

       #总结
        for i in range(len(a)):
            zhi[i] = a[i]
        #print("```````````````````````````5")

        zhi = lvbo(zhi, 55, 150,fs)

        zhi = list(zhi)
        #for i in range(500):
            #fang.append(0)
        p = np.array(zhi)

        # 多进程计算心率开始
        print("`````````````````````````````````````zhi", zhi)
        #print("`````````````````````````````````````p", p)
    return picture,picture2,picture3,p


def get_origial(cnt_arry, L, cn, cn2, cn3,H, H2,H3):
    if (cnt_arry > L):
        # H.append(cn)
        #print("``````````````````````cn", cn)
        H2.append(cn2)
        H.append(cn)
        H3.append(cn3)


    return H,H2,H3


def fs_contral(time_s_n, t):
    while ((1 / (time.time() - time_s_n)) > t):
        for i in range(5):
            i = i  # 控制fs为4


def fs_mean(cnt_arry, L, cnt_fs, time_n):
    if (cnt_arry > L):  # 当数组中有第一个数时开始计算
        if (cnt_fs == 0):
            cnt_fs = cnt_fs + 1
            time_n = time.time()
        elif (cnt_fs < 50):
            cnt_fs = cnt_fs + 1
        elif (cnt_fs == 50):
            cnt_fs = 0  # 每隔100帧刷新一次fs
            fs.value = 50 / (time.time() - time_n)
            #print('the zai limian  fs is ', (fs.value), '!!!!!!!!!!!!!!!!!!!!!!!!!!')
            time_n = time.time()
    return time_n, cnt_fs


# 小波函数
def pywav(data):
    # Create wavelet object and define parameters
    w = pywt.Wavelet('db12')  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    print("maximum level is " + str(maxlev))

    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(data, 'db12', level=maxlev)  # 将信号进行小波分解
    for i in range(1, len(coeffs)):  # coeffs为小波分解后的列表
        tmp = coeffs[i].copy()
        Sum = 0.0
        for j in coeffs[i]:
            Sum = Sum + abs(j)
        N = len(coeffs[i])
        Sum = (1.0 / float(N)) * Sum
        sigma = (1 / 0.6745) * Sum
    lamda = sigma * math.sqrt(2.0 * math.log(float(N), math.e))  # lamda为求出的阈值

    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], lamda)  # 将噪声滤波

    datarec = pywt.waverec(coeffs, 'db12')  # 将信号进行小波重构
    return datarec

from scipy import signal

def lvbo(timeseries,min,max,fs):
    b, a = signal.butter(8, [2 * min / 60 / fs, 2 * max / 60 / fs], 'bandpass')  # 配置滤波器 8 表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, timeseries)  # data为要过滤的信号
    return filtedData


# 画绿色通道的图像
# def plotGreen(n):

def defg(x):
    global n3
    global l3
    if ((x.event_type == "down") and (x.name == "5")):
        n3 += 1
    if ((x.event_type == "down") and (x.name == "6")):
        n3 -= 1
    if ((x.event_type == "down") and (x.name == "7")):
        l3 += 1
    if ((x.event_type == "down") and (x.name == "8")):
        l3 -= 1
    return n3, l3
from ICA.ICAcode import *
def cont_normal(cnt_frame,H, H2,H3,flag1,fs):
    p = []
    p2 = []
    C = []
    if ((cnt_frame >= 210)):
        # 多进程共享数值初始化
        # ································求出均值与方差··························
        if len(H)>210:
            H = H[20:210]
        if len(H2)>210:
            H2 = H2[20:210]
        if len(H3)>210:
            H3 = H3[20:210]

        X = np.empty((len(H), 3))

        X[:, 0] = H
        X[:, 1] = H2
        X[:, 2] = H3

        x = 3*X[:,2] - 2*X[:,1]
        Y = 1.5*X[:,2] +X[:,1] -1.5*X[:,1]
        x = lvbo(x, 55, 150,fs)
        Y = lvbo(Y, 55, 150,fs)

        x_std = np.std(x, ddof=1)
        Y_std = np.std(Y, ddof=1)
        C = x - (x_std/Y_std)*Y
        C = np.array(C)
        C = lvbo(C, 55, 150,fs)
        #p  = ica_dim(x)
        p = np.array(p)

        p2 = H2.copy()
        p2 = np.array(p2)
        p2 = lvbo(p2, 55, 150, fs)


        # 多进程计算心率开始

    return C,p2,flag1,H,H2,H3
"""from PAC.SQL import *
import datetime
timeP = ['Morning','Afternoon','Night']
db = pymysql.connect(host="175.24.51.143", port=3306, user="projectTest", password="123456", database="test",
                     charset="utf8")
PULS1 = [15, 14, 13, 15, 16, 15, 15, 14, 12, 11, 13, 12]
tem = [36.1, 36.2, 36.1, 36.1, 36.3, 36.1, 36.1, 36.2, 36.2, 36.2, 36.1, 36.3]"""
def send_mysql(PULS,i):
    if (datetime.datetime.now().hour < 15 and datetime.datetime.now().hour > 11):  # Midday datas upload
        insertMid([int(PULS), int(PULS1[i]), tem[i], 'Guo'], timeP, times, db)
        print(PULS)
    elif (datetime.datetime.now().hour < 6 and datetime.datetime.now().hour >= 15):  # Night datas update
        insertNig([int(PULS), int(PULS1[i]), tem[i], 'Guo'], timeP, times, db)
        print(PULS)
    else:
        insertMor([int(PULS), int(PULS1[i]), tem[i], 'Guo'], timeP, times, db)
        print(PULS)


# 计算心率,直方图方法
def cont_zhi(cnt_frame,picture,zhi,changdu,fs):
    #print("``````````````````face", len(face))
    p = []
    if ( (cnt_frame>150)):
    # ································求出均值与方差··························
        if (len(picture[1])>210):
            for i in range(changdu):
                picture[i] = picture[i][20:150]

        #算鼻子部分
        #print("```````````````````````````1",len(picture[i]))
        zhi = ([0] * (len(picture[1]) -1))
        a = ([0] * (len(picture[1]) -1)) # 多少个时刻，应该与picture中最短的时刻相匹配
        b = ([0] * (len(picture[1]) -1 ))
        #print("```````````````````````````2")
        for i in range(changdu):  #循环每一行
            data = picture[i].copy()
            data = lvbo(data, 55, 80,fs)
            diff_x1 = np.diff(data)
            cn = 0
            for j in range(len(diff_x1)):  #第i行的第j个时刻
                if diff_x1[j] > 0:             #最后一行在上一个时刻没有出现，所以其j少一个，即其不为a[Tmax]贡献
                    a[j]+=1

                else:
                    for k in range(8):
                        if diff_x1[j - k] < 0:
                            cn += 1
                            #print("````````````````````````",cn)
                    if cn >= 400:
                        a[j] += 0
                        print("````````````````````````")
                    else:
                        a[j] += 0
                    cn = 0

        #print("```````````````````````````3")


       #总结
        for i in range(len(a)):
            if abs(a[i]) > abs(b[i]):
                zhi[i] = a[i]
            else:
                zhi[i] = b[i]
        #print("```````````````````````````5")

        fang = zhi.copy()

        fang = lvbo(fang, 55, 80,fs)

        fang = list(fang)
        for i in range(500):
            fang.append(0)
        p = np.array(fang)


        # 多进程计算心率开始
        #print("`````````````````````````````````````zhi", zhi)
        #print("`````````````````````````````````````p", p)
        #print("`````````````````````````````````````p2", H2)
        #flag1.value = 0
        #print("```````````````````````````````start")
    return picture,zhi,p







def result(puls_n,puls_n2,puls,puls2,PULS,PULS1):
    if (puls_n != 0):
        if (abs(puls_n - puls.value) > 10):
            puls_n = puls_n
        elif (puls.value < 70):
            puls_n = puls_n
        else:
            puls_n = puls.value
    else:
        puls_n = puls.value

    if (puls_n2 != 0):
        if (abs(puls_n2 - puls2.value) > 10):
            puls_n2 = puls2.value
        elif (puls2.value < 70):
            puls_n2 = puls2.value
        else:
            puls_n2 = puls2.value
    else:
        puls_n2 = puls2.value

    if puls_n2 and puls_n:
        PULS.append(puls_n)
        PULS1.append(puls_n2)

    return puls_n,puls_n2,PULS,PULS1

def fs_cont(time_s_n,time_mean,cnt_fs_mean,cnt_arry, L):
    return time_s_n,time_mean,cnt_fs_mean


def begin2(capture, puls_n, puls_n2,rect_face,cnt_arry):
    # ···············变量初始化(其中部分变量未使用到)·················
    global L
    L = 60

    # 变量定义
    if (L == 60):
        global n2
        global n
        global start
        global end
        global i_face
        global M
        global F
        global n3
        global l3
        global PULS
        global PULS1
        n3 = 3
        l3 = 3
        U1 = 3
        U2 = 3

        M = 0
        F = []
        face[:] = []
        H = []
        H2 = []
        H3 = []
        cnt = 0
        U_face = []
        start = 0
        i_face = 0
        rect_face_n = []
        flag_roi = False  # 初始化flag_roi为0
        roi = np.array([0, 0, 0, 0], dtype='int32')
        roi2 = np.array([0, 0, 0, 0], dtype='int32')
        roi_cnt = np.array([0, 0, 0, 0], dtype='int32')
        roi_cnt2 = np.array([0, 0, 0, 0], dtype='int32')

        roi_U1 = np.array([0, 0, 0, 0], dtype='int32')
        roi_U2 = np.array([0, 0, 0, 0], dtype='int32')
        roi_cent_U1 = np.array([0, 0, 0, 0], dtype='int32')
        roi_cent_U2 = np.array([0, 0, 0, 0], dtype='int32')

        flag2 = 0
        flag3 = 0
        flag3 = 0
        cnt_3 = 0
        time_n = 0
        cnt_fs = 0
        cnt_fs0 = 0
        b = 0
        fs_s = 12
        cnt_4 = L / fs_s
        fs_t = 0
        time_s_n = 0
        rect_face_n = [0] * (10000)
        rect_face_n2 = [0] * (10000)
        rect_face_u1 = [0] * (10000)
        rect_face_u2 = [0] * (10000)
        rect_face_nu1 = [0] * (10000)
        rect_face_nu2 = [0] * (10000)
        img_n = [0] * (10000)

        rect_face_n5 = [0] * (10000)
        rect_face_n6 = [0] * (10000)
        rect_face_n7 = [0] * (10000)
        flag4 = 1
        draw_ax(PULS, PULS1)
        PULS = []
        PULS1 = []
        cnt3 = 0
        PULSZ = []
        PULSZ1 = []
        xinshi = 0
        flag_kal = 0
        cnt_kal = 0
        puls_skip = []
        cnt_skip = 0
        changdu = 0
        zhi = []
        picture = []


    # ````````````获取像素值变化``````````````
    picture,zhi,changdu = get_vary(changdu,zhi,picture,cnt_arry,L,rect_face)

    # 给予被测者一段稳定时间后开始获取原始序列
    H,H2,H3 = get_origial(cnt_arry, L, cn, cn2, cn3,H, H2,H3)

    # ---------------------计算心率---------------------
    cont_normal(H, H2,H3)

    puls_n,puls_n2 = result(puls_n,puls_n2)

