
# coding: utf-8

# In[ ]:


from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *
base = BaseOverlay("base.bit")
import numpy as np


# In[ ]:


# monitor configuration: 640*480 @ 60Hz
Mode = VideoMode(640,480,24)
hdmi_out = base.video.hdmi_out
hdmi_out.configure(Mode,PIXEL_BGR)
hdmi_out.start()
frame_out_w = 320
frame_out_h = 240
# camera (input) configuration
frame_in_w = 320
frame_in_h = 240


# In[ ]:


def hdmi_show(frame,hdmi_out):
    outframe = hdmi_out.newframe()#建立一个用于存放输出画面的参数空间，三维数组，其尺寸有hdmi_out决定，hdmi_out由前面HDMI初始模块确定
    #print("ori",type(outframe),len(outframe))
    #outframe[120:120+frame_out_h,160:160+frame_out_w,:] = frame[120:frame_out_h+120,0:frame_out_w,:]#将画面数据格式转换为HDMI能够读取的格式，注意通道顺序
    outframe[:,:,:] = frame[:,:,:]#将画面数据格式转换为HDMI能够读取的格式，注意通道顺序
    
    #print("set",type(frame_vga),len(frame_vga))
    hdmi_out.writeframe(outframe)


# In[ ]:


heart_list = np.random.randint(65,70,size=200)
heart_list


# In[ ]:


#!/usr/bin/python
# -*-coding:utf-8 -*-

import socket
import cv2
import numpy

# 接受图片大小的信息
def recv_size(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)   #最大接受count个字节
        #print(newbuf)
        if not newbuf: return None
        buf += newbuf      #buf中加上接受的字符的内容
        count -= len(newbuf)        #count减去字节的长度
    return buf




# socket.AF_INET 用于服务器与服务器之间的网络通信
#socket.SOCK_STREAM 代表基于TCP的流式socket通信
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 设置地址与端口，如果是接收任意ip对本服务器的连接，地址栏可空，但端口必须设置
address = ('192.168.0.102', 8888)
s.bind(address)  # 将Socket（套接字）绑定到地址
s.listen(True)   # 开始监听TCP传入连接
print ('Waiting for images...')

# 接受TCP链接并返回（conn, addr），其中conn是新的套接字对象，可以用来接收和发送数据，addr是链接客户端的地址。

conn, addr = s.accept()


###############################################

import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont

# 心跳数据
t_size = 1000
heart_list = np.random.randint(65,70,size=1000)
list_index = 0

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
#     fontStyle = ImageFont.truetype(
#         "font/simsun.ttc", textSize, encoding="utf-8")
#     # 绘制文本
#     draw.text((left, top), text, textColor, font=fontStyle)
#     # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
scale = 0
# print('cap')
# cap = cv2.VideoCapture(0)
# print('cap2')
# # Check if camera opened successfully
# if (cap.isOpened() == False):
#     print("Error opening video stream or file")
frameNum = 0
# Read until video is completed
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
W =[]
cnt = 0
cnt2 = 0
cnt2_n = 0
cnt3= 0
flag =0
cnt4 = 0
################################################
while True:
    length = recv_size(conn,16)  # 首先接收来自客户端发送的大小信息
    length = length.decode()
    if isinstance(length, str):  # 若成功接收到大小信息，进一步再接收整张图片
        #print("length",length)
        stringData = recv_size(conn,int(length))
        data =numpy.fromstring(stringData, dtype='uint8')
        decimg = cv2.imdecode(data,1)  # 解码处理，返回mat图片
        
         #
        #print('Image recieved successfully!')
        
############################################################
        #frame = cv2.resize(frame, (int(a[1] / 3), int(a[0] / 3)), interpolation=cv2.INTER_CUBIC)
        frame = decimg.copy()
        frameNum += 1
        time1 = time.time()
        tempframe = frame
        if (frameNum == 1):
            previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)
        if (frameNum >= 2):
            time0 = time.time()

            currentframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)

            currentframe = cv2.absdiff(currentframe, previousframe)
            #currentframe = currentframe+currentframe2

            clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(10, 10))
            currentframe = clahe.apply(currentframe)

            median = cv2.medianBlur(currentframe, 3)

            #        img = cv2.imread("E:/chinese_ocr-master/4.png")
            #        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, threshold_frame = cv2.threshold(median, 10, 255, cv2.THRESH_OTSU)
            gauss_image = cv2.GaussianBlur(threshold_frame, (3, 3), 0)


            edge_output = cv2.Canny(gauss_image, 50, 150)


            #fgmk = cv2.morphologyEx(edge_output, cv2.MORPH_CLOSE, kernel)



            # Display the resulting frame

            image = frame.copy()
            #fgmk = cv2.medianBlur(fgmk, 3)
            # 查找轮廓
            contours = cv2.findContours(edge_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in contours[1]:
                if cv2.contourArea(c) > 1500 and cv2.contourArea(c) < 15000:
                    (x, y, w, h) = cv2.boundingRect(c)
                    if cnt<2:

                        W.append(y)
                        cnt +=1
                    else:
                        W.append(y)
                        del(W[0])

                    if scale == 0: scale = -1;break
                    scale = w / h

                    cv2.putText(image, "scale:{:.3f}".format(scale), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)
                    cv2.drawContours(image, [c], -1, (255, 0, 0), 1)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    image = cv2.fillPoly(image, [c], (255, 255,
                                                      255))  # 填充
                    flag = 1
            # 根据image人体比例判断
            if scale > 0.75:
                #print(W)

                if (  y - W[0])>30  and y!=0:
                    #print(cnt2)
                    #print(W)
                    if cnt2 > 1:
                        cv2.putText(image, "     Falled", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
                    elif cnt2>0:
                        cv2.putText(image, "     Walking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
                        
                        cv2.putText(image, "     Heart: {}".format(heart_list[list_index]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
                        print("walking")
                    if flag:
                        cnt2+=1

                elif cnt2>2:
                    cv2.putText(image, "     Falled", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
                else:
                    cv2.putText(image, "     Walking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
                    #cv2.putText(image, "     Heart: {}".format(heart_list[list_index]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
                    print("walking")
                flag = 0

                # cv2.putText(img, "Walking 行走中", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)#行走中
            #if scale > 0.9 and scale < 2:
                #image = cv2ImgAddText(image, "Falling 中间过程", 10, 20, (255, 0, 0), 30)  # 跌倒中
                # cv2.putText(img, "Falling 跌倒中", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)#跌倒中
            if scale < 0.75 and scale > 0:
                if cnt2 < 2:
                    cv2.putText(image, "     Walking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
                    print("walking")
                    cv2.putText(image, "     Heart: {}".format(heart_list[list_index]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
                elif cnt2 >3:
                    cv2.putText(image, "     Walking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
                    print("walking")
                    cv2.putText(image, "     Heart: {}".format(heart_list[list_index]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
                    cnt2 = 0
                else:
                    cv2.putText(image, "     Falled", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)

            cnt3 += 1
            if cnt3 ==3:
                cnt3 = 0
                if cnt2 == cnt2_n and cnt2<5:
                    cnt2 = cnt2
                cnt2_n = cnt2

            hdmi_show(image,hdmi_out)
            print(1/abs(time.time()-time0))

                    # Press Q on keyboard to  exit
        c = cv2.waitKey(1)  # 停止
        list_index +=1
        if (c == 27):
            break
        if (base.buttons[3].read()==1):
            break
        if (base.buttons[0].read()==1):
            flag = 1
            #breathe.value = 0
            break

        previousframe = cv2.cvtColor(tempframe, cv2.COLOR_BGR2GRAY)


# When everything done, release the video capture object
# cap.release()
# hdmi_out.stop()
# del hdmi_out
# Closes all the frames
##############################################################
        
        
        
        
        
        
        
    c = cv2.waitKey(1)  # 停止\
    if (c == 27):
        break
s.close()
cv2.destroyAllWindows()

