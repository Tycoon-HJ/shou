```
import cv2
import numpy as np
import copy
import math
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains

# 加载chromedriver驱动
dr = webdriver.Chrome(executable_path='C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe')
# 放大谷歌浏览器窗口
dr.maximize_window()
#
web_path = r'file:///C:\Users\25232\Desktop\新建文件夹\index.html'
dr.get(web_path)
dr.maximize_window()
# parameters
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 60  # BINARY threshold
blurValue = 41  # GaussianBlur parameter
# 背景阈值
bgSubThreshold = 50
learningRate = 0

# 检测到手的点的临时坐标
x_position_temp = -1
# 帧计数器
x_position_count = 0
# 刷新检测到点的位置
x_position_trigger = False

# variables
isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyborad simulator works


# 屏幕点击函数
def click_locxy(dr, x, y, left_click=True):
    if left_click:
        ActionChains(dr).move_by_offset(x, y).click().perform()
    else:
        ActionChains(dr).move_by_offset(x, y).context_click().perform()
    ActionChains(dr).move_by_offset(-x, -y).perform()  # 将鼠标位置恢复到移动前


# 阈值的改动
# def printThreshold(thr):
#     print("! Changed threshold to " + str(thr))

# 移除与上一次相同的背景，返回前景的数据
def removeBG(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res


# 检测手指个数函数
def calculateFingers(res, drawing):  # -> finished Boolean值, cnt: 手指凸包点
    # 凸性缺陷
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # 避免碰撞.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # 计算角度
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # 余弦定理

                if angle <= math.pi / 2:  # 角度小于90度，视为手指
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            # print(res[defects[0][0][0]][0][0],res[defects[0][0][0]][0][1])
            # 是否有手指，手指个数-1，第一个检测点的x坐标
            return True, cnt, res[defects[0][0][0]][0][0]
    return False, 0, -1


# 摄像头
camera = cv2.VideoCapture(0)
camera.set(10, 200)
# cv2.namedWindow('trackbar')
# cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold)


while camera.isOpened():
    # frame帧
    ret, frame = camera.read()
    # threshold = cv2.getTrackbarPos('trh1', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # 过滤器
    frame = cv2.flip(frame, 1)  # 水平翻转框架

    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)

    #  Main operation
    if isBgCaptured == 1:  # 在捕获背景之前不会运行
        img = removeBG(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
              int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # 剪裁获取到的背景
        # cv2.imshow('mask', img)

        # 将图像转换为二值图像
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        # cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        # cv2.imshow('ori', thresh)

        # 获取coutours
        thresh1 = copy.deepcopy(thresh)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # 找到最大轮廓（根据面积）
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            isFinishCal, cnt, x_position = calculateFingers(res, drawing)
            x_position_count += 1
            # if x_position_count % 30 == 0:
            #     if x_position > x_position_temp:
            #         print("%d: 向右" % x_position_count)
            #     elif x_position < x_position_temp:
            #         print("%d: 向左" % x_position_count)
            #     elif x_position == 0:
            #         continue
            #     else:
            #         pass
            #     x_position_temp = x_position
            if x_position_count % 60 == 0:
                if x_position != -1:
                    if not x_position_trigger:
                        x_position_temp = x_position
                        x_position_count += 55
                    x_position_trigger = True
                else:
                    x_position = False
            if x_position_trigger == True:
                if x_position_count % 60 == 0:
                    if x_position > x_position_temp:
                        print("%d: 向右" % x_position_count)
                        click_locxy(dr, 285 - 0, 438 - 70)  # 左键点击
                        # ActionChains(dr).move_by_offset(1436,347).click().perform()
                        # dr.find_element_by_class_name("nextBtn").click()
                    elif x_position < x_position_temp:
                        print("%d: 向左" % x_position_count)
                        # dr.find_element_by_class_name("lastBtn").click()
                        # ActionChains(dr).move_by_offset(470,347).context_click().perform()
                        # ActionChains(dr).move_by_offset(470,347).click().perform()
                        click_locxy(dr, 1250 - 0, 435 - 70)  # 左键点击
                    else:
                        pass
                    x_position_temp = x_position

            if triggerSwitch is True:
                if isFinishCal is True and cnt <= 2:
                    print(cnt)
                    # app('System Events').keystroke(' ')  # simulate pressing blank space

        # cv2.imshow('output', drawing)

    # 键盘操作
    k = cv2.waitKey(10)
    if k == 27:  # 按esc退出
        camera.release()
        cv2.destroyAllWindows()
        break
    elif k == ord('b'):  # 按b获取当前背景
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        print('!!!Background Captured!!!')
    elif k == ord('r'):  # 按r重新获取当前背景
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print('!!!Reset BackGround!!!')
    elif k == ord('n'):  # 按n触发开关
        triggerSwitch = True
        print('!!!Trigger On!!!')


```
