import numpy as np

from stgcn.stgcn import STGCN
import time

POSE_MAPPING = ["站着", "走着", "坐着", "躺下", "站起来", "坐下", "摔倒"]

POSE_MAPPING_COLOR = [
    (255, 255, 240), (245, 222, 179), (244, 164, 96), (210, 180, 140),
    (255, 127, 80), (255, 165, 79), (255, 48, 48)
]

# 为了检测动作的准确度，每30帧进行一次检测
ACTION_MODEL_MAX_FRAMES = 30

action_model = STGCN(weight_file='./weights/tsstg-model.pth', device='cpu')


# self.action_model = STGCN(weight_file='./weights/checkpoint_iter_370000.pth', device='cpu')
# self.joints_list = deque(maxlen=ACTION_MODEL_MAX_FRAMES)

def sleep(result):
    response = {
        'first_sleep_time': None,
        'sleep_end_time': None,
        'sleep_duration': None,
    }
    first_sleep_time = 0
    if result == '躺下':  # 如果识别结果为Sleep
        first_sleep_time = time.time()
        sleep_begin_time = time.strftime('%Y/%m/%d  %H:%M', time.localtime())
        # print(f"Sleep detected at ", sleep_begin_time)
    elif result == '坐着' and first_sleep_time:
        if result == '坐着':  # 如果识别结果为GetUp
            sleep_end_time = time.strftime('%Y/%m/%d  %H:%M', time.localtime())
            # print(f"Get up detected at", sleep_end_time)
            if first_sleep_time and sleep_end_time:  # 如果之前有记录睡眠时间
                sleep_times = (time.time() - first_sleep_time) / 60
                print(f"Sleep duration: {sleep_times} minutes")
                if sleep_times:
                    return {
                        'first_sleep_time': first_sleep_time,
                        'sleep_end_time': sleep_end_time,
                        'sleep_duration': sleep_times,
                    }

# 识别动作
def detection(joints_list, image_w, image_h):
    action = ''
    # 30帧数据预测动作类型
    if len(joints_list) == ACTION_MODEL_MAX_FRAMES:
        pts = np.array(joints_list, dtype=np.float32)
        out = action_model.predict(pts, (image_w, image_h))

        index = out[0].argmax()
        action_name = POSE_MAPPING[index]
        cls = POSE_MAPPING_COLOR[index]
        action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
        print(action)
        # if sleep(action_name):
        #     print(sleep(action_name))
    return action
