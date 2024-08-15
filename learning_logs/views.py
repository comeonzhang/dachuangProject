from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render
from .models import SleepRecord, EatRecord, User
from django.http import Http404

import json
from django.views.decorators.csrf import csrf_exempt

import socket
import numpy as np

import ast
from collections import deque
from learning_logs.code.stgcn.stgcn import STGCN
import time
from django.utils import timezone

# Create your views here.
POSE_MAPPING = ["站着", "走着", "坐着", "躺下", "站起来", "坐下", "摔倒"]

POSE_MAPPING_COLOR = [
    (255, 255, 240), (245, 222, 179), (244, 164, 96), (210, 180, 140),
    (255, 127, 80), (255, 165, 79), (255, 48, 48)
]

# 为了检测动作的准确度，每30帧进行一次检测
ACTION_MODEL_MAX_FRAMES = 30
first_sleep_time = 0
sleep_begin_time = ''

# action_model = STGCN(weight_file='code/weights/tsstg-model.pth', device='cpu')
action_model = STGCN(weight_file='learning_logs/code/weights/tsstg-model.pth', device='cpu')


# self.action_model = STGCN(weight_file='./weights/checkpoint_iter_370000.pth', device='cpu')
# self.joints_list = deque(maxlen=ACTION_MODEL_MAX_FRAMES)


def eat(result, th=15):
    first_eat_time = time.time()
    eat_begin_time = time.strftime('%Y/%m/%d  %H:%M', time.localtime(first_eat_time))
    print(f"Start eating at", eat_begin_time)

    if result == 'Eat':
        time_counter = (first_eat_time - time.time()) / 60
        t = th * 60
        if time_counter > t:  # 判断是否超过15分钟
            eat_end_time = time.strftime('%Y/%m/%d  %H:%M', time.localtime(time.time()))
            print(f"Stop eating at", eat_end_time)
            print(f"Eating duration: {time_counter} seconds")
            return {'eat_times': time_counter,
                    'eat_begin_time': eat_begin_time,
                    'eat_end_time': eat_end_time,
                    }


def detection(joints_list, image_w, image_h, owner):
    action = ''
    global first_sleep_time
    global sleep_begin_time
    data = {'owner': User.objects.get(id=owner),
            'sleep_times': None,
            'sleep_begin_time': sleep_begin_time,
            'sleep_end_time': None,
            }
    # 30帧数据预测动作类型
    # print(len(joints_list))
    if len(joints_list) == ACTION_MODEL_MAX_FRAMES:
        # pts = np.array(joints_list, dtype=np.float32)
        out = action_model.predict(joints_list, (image_w, image_h))
        index = out[0].argmax()
        action_name = POSE_MAPPING[index]
        # cls = POSE_MAPPING_COLOR[index]
        action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
        # print(action)
        # if action_name == "摔倒":
        #     print("Detected a fall. Sending alert.")
        #     send_date = 1
        #     # 停止检测5秒
        #     time.sleep(5)
        # if action_name == "睡觉":
        if action_name == '躺下':  # 如果识别结果为Sleep
            if not first_sleep_time:
                first_sleep_time = time.time()
                sleep_begin_time = time.strftime('%Y/%m/%d  %H:%M', time.localtime())
                # print(f"Sleep detected at ", sleep_begin_time)
                print(data)
        elif action_name == '坐着' and first_sleep_time:
            data['sleep_end_time'] = time.strftime('%Y/%m/%d  %H:%M', time.localtime())
            # print(f"Get up detected at", sleep_end_time)
            # if sleep_end_time:  # 如果之前有记录睡眠时间
            print(data)
            data['sleep_times'] = (time.time() - first_sleep_time) / 60
            print(f"Sleep duration: {data['sleep_times']} minutes")
            first_sleep_time = 0
            print(data)
            SleepRecord.objects.create(**data)
        # elif action_name == "吃饭":
        #     eat(action_name)
    return action


def index(request):
    return render(request, 'learning_logs/index.html')


@login_required
def Information(request):  # Django从服务器收到request对象
    """
    # infors = Infor.objects.order_by('date_added')#查询数据库，请求提供Topic对象，date_added排序，返回查询集
    infors = Infor.objects.filter(owner=request.user).order_by('date_added')  # 只从数据库获取owner属性为当前用户Topic对象

    # if infors.owner != request.user:
    #     raise Http404
    context = {'infors': infors}  # 定义一个将发送给模板的上下文（字典），一组显示的主题（键：模板中访问数据的名称，值：发送给模板的数据）
    return render(request, 'learning_logs/infors.html', context)  # 对象，模板路径，变量(字典)"""
    # 获取今天日期，并将时间设置为午夜（一天的开始）
    today = timezone.localdate()

    # 获取今天前7天的日期列表（不包括今天）
    date_range = [today - timezone.timedelta(days=x) for x in range(6, -1, -1)]

    # 准备上下文字典，用于传递到模板
    context = {
        'daily_activities_by_date': []
    }

    # 遍历日期范围
    for single_date in date_range:
        # 对每个日期获取睡眠和饮食记录
        sleep_records = SleepRecord.objects.filter(owner=request.user, date_added=single_date)
        eat_records = EatRecord.objects.filter(owner=request.user, date_added=single_date)

        # 将记录添加到上下文字典
        context['daily_activities_by_date'].append({
            'date': single_date,
            'sleep_records': sleep_records,
            'eat_records': eat_records,
        })

    return render(request, 'learning_logs/infors.html', context)


@csrf_exempt  # 禁用CSRF令牌，因为API来自其他服务器
def receive_data(request):
    global my_deque
    if request.method == 'POST':
        data = json.loads(request.body)
        # 这里处理接收到的数据，例如保存到数据库
        # for item in data:
        # BehaviorData.objects.create(**data)
        joints_list = data.get('data')
        owner = data.get('owner_id')
        # print(data.get('3'))
        # print(type(joints_list))
        try:
            parsed_data = ast.literal_eval(joints_list)
        except ValueError as e:
            print(f"Error parsing the string: {e}")
            # 处理错误或返回错误响应
        else:
            # 将解析后的数据转换为 deque
            my_deque = np.array(parsed_data)
            # print(my_deque)
        image_w = data.get('image_w')
        image_h = data.get('image_h')
        action = detection(my_deque, image_w, image_h, owner)
        return JsonResponse({'status': 'success'}, safe=False)
    return JsonResponse({'status': 'error'}, safe=False)


def tcp_server(request):
    # 设置服务器的IP和端口
    HOST = '127.0.0.1'  # 或者使用服务器的IP地址
    PORT = 8712

    # 创建socket对象
    def listen_for_data():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen()

            while True:
                # 接受客户端的连接
                conn, addr = s.accept()
                with conn:
                    print('Connected by', addr)
                    while True:
                        # 接收数据
                        data = conn.recv(1024)
                        if not data:
                            break
                        # 处理数据（这里只是打印出来）
                        print('Received:', data.decode())
                        # 可以在这里调用视图函数处理数据并返回响应
    # server_thread = threading.Thread(target=listen_for_data)
    # server_thread.daemon = True  # 设置为守护线程，以便在主程序退出时自动关闭
    # server_thread.start()
