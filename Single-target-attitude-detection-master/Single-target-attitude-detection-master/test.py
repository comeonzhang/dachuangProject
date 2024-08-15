import cv2
import mediapipe as mp
import time

def motion_detection(Js, Je, Tp=8, Ts=0.5, sh=5):
    a = 0
    # 检查关节点置信度
    for i in range(min(len(Js), len(Je))):
        Ss, Se = Js[i][2], Je[i][2]  # 获取置信度
        # if Ss and Se == 0:
        #     print(i)
        #     continue
        # print(Ss)
        # print(Se)
        if (Ss > Ts and Se < Ts) or (Ss < Ts and Se > Ts):
            # 同一关节点消失或出现
            a += 1
        elif (Ss > Ts and Se > Ts):
            # 同一关节点出现
            Xs, Ys, Xe, Ye = Js[i][0], Js[i][1], Je[i][0], Je[i][1]
            if abs(Xs - Xe) > Tp or abs(Ys - Ye) > Tp:
                # 坐标变化超过偏移量阈值
                a += 1
    print(a)
    if a > sh:
        return True
    else:
        return False

# 导入姿态跟踪方法
mpPose = mp.solutions.pose  # 姿态识别方法
pose = mpPose.Pose(static_image_mode=False,  # 静态图模式，False代表置信度高时继续跟踪，True代表实时跟踪检测新的结果
                   # upper_body_only=False,  # 是否只检测上半身
                   smooth_landmarks=True,  # 平滑，一般为True
                   min_detection_confidence=0.5,  # 检测置信度
                   min_tracking_confidence=0.5)  # 跟踪置信度
# 检测置信度大于0.5代表检测到了，若此时跟踪置信度大于0.5就继续跟踪，小于就沿用上一次，避免一次又一次重复使用模型

# 导入绘图方法
mpDraw = mp.solutions.drawing_utils

# （1）导入视频
filepath = 0
cap = cv2.VideoCapture(filepath)
#cap.set(cv2.CAP_PROP_FPS, 10)#帧率 帧/秒
fps2 = cap.get(cv2.CAP_PROP_FPS)

pTime = 0  # 设置第一帧开始处理的起始时间

# （2）处理每一帧图像
lmlist = []  # 存放人体关键点信息
lmlist1 = []  # 存放人体关键点信息
#lmlist2 = []  # 存放人体关键点信息
count = 0

while True:

    # 接收图片是否导入成功、帧图像
    success, img = cap.read()

    # 将导入的BGR格式图像转为RGB格式
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 将图像传给姿态识别模型
    results = pose.process(imgRGB)

    # 查看体态关键点坐标，返回x,y,z,visibility
    # print(results.pose_landmarks)

    # 如果检测到体态就执行下面内容，没检测到就不执行
    if results.pose_landmarks:

        # 绘制姿态坐标点，img为画板，传入姿态点坐标，坐标连线
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # 获取32个人体关键点坐标, index记录是第几个关键点
        for index, lm in enumerate(results.pose_landmarks.landmark):
            # 保存每帧图像的宽、高、通道数
            h, w, c = img.shape

            # 得到的关键点坐标x/y/z/visibility都是比例坐标，在[0,1]之间
            # 转换为像素坐标(cx,cy)，图像的实际长宽乘以比例，像素坐标一定是整数
            cx, cy, cv = int(lm.x * w), int(lm.y * h), lm.visibility
            if cv < 0.5:
                cv = 0

            # 打印坐标信息
            #print(index, cx, cy, cv)

            # 保存坐标信息
            lmlist.append((cx, cy, cv))

            # 在关键点上画圆圈，img画板，以(cx,cy)为圆心，半径5，颜色绿色，填充圆圈
            cv2.circle(img, (cx, cy), 3, (0, 255, 0), cv2.FILLED)

    # 查看FPS
    cTime = time.time()  # 处理完一帧图像的时间
    fps = 1 / (cTime - pTime)
    pTime = cTime  # 重置起始时间

    # 在视频上显示fps信息，先转换成整数再变成字符串形式，文本显示坐标，文本字体，文本大小
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(img, str(int(fps2)), (70, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    # 调用函数检测运动
    if count == 0 or count == 30:
        if count == 0:
            lmlist1 = list(lmlist)
            # print('AAAAAAAA', count, index, cx, cy, cv)
        elif count == 30:
            #lmlist2 = list(lmlist)
            # print(count, index, cx, cy, cv)
            count = -1
            motion = motion_detection(lmlist1, lmlist)
            print("Motion detected:", motion)

    count += 1
    lmlist.clear()

    # 显示图像，输入窗口名及图像数据
    cv2.imshow('image', img)
    if cv2.waitKey(10) & 0xFF == 27:  # 每帧滞留15毫秒后消失，ESC键退出
        break

# 释放视频资源
cap.release()
cv2.destroyAllWindows()