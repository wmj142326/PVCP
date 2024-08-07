import numpy as np
import os, sys
import pickle
import yaml
from easydict import EasyDict as edict
from typing import Any, IO

import os.path as osp
import numpy as np
import argparse
import joblib
import json
import cv2
import shutil  
import imageio
import matplotlib.pyplot as plt  


ROOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')

class TextLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        with open(self.log_path, "w") as f:
            f.write("")
    def log(self, log):
        with open(self.log_path, "a+") as f:
            f.write(log + "\n")

class Loader(yaml.SafeLoader):
    """YAML Loader with `!include` constructor."""

    def __init__(self, stream: IO) -> None:
        """Initialise Loader."""

        try:
            self._root = os.path.split(stream.name)[0]
        except AttributeError:
            self._root = os.path.curdir

        super().__init__(stream)

def construct_include(loader: Loader, node: yaml.Node) -> Any:
    """Include file referenced at node."""

    filename = os.path.abspath(os.path.join(loader._root, loader.construct_scalar(node)))
    extension = os.path.splitext(filename)[1].lstrip('.')

    with open(filename, 'r') as f:
        if extension in ('yaml', 'yml'):
            return yaml.load(f, Loader)
        elif extension in ('json', ):
            return json.load(f)
        else:
            return ''.join(f.readlines())

def get_config(config_path):
    yaml.add_constructor('!include', construct_include, Loader)
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=Loader)
    config = edict(config)
    _, config_filename = os.path.split(config_path)
    config_name, _ = os.path.splitext(config_filename)
    config.name = config_name
    return config

def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
        
def read_pkl(data_url):
    file = open(data_url,'rb')
    content = joblib.load(file)
    file.close()
    return content



def get_idx_groups(dictionary):
    idx_groups = {}
    for key, value in dictionary.items():
        idx = value.get("idx")
        if idx is not None:
            if idx not in idx_groups:
                idx_groups[idx] = []
            idx_groups[idx].append((key, value))
    return idx_groups

def effective_data(src_data):
    dis_data = {}
    for frame, label in src_data.items():
        if src_data[frame]["keypoint_2d"]:
            dis_data[frame] = src_data[frame]
    return dis_data

def get_idx_frame_list(idx_groups, idx):
    idx_frame_list = [entry[0] for entry in idx_groups[idx]]
    return idx_frame_list

def find_longest_consecutive_subset_with_gap(nums, gap):
    if not nums or gap < 0:
        return [], []

    num_to_idx = {}
    for idx, num in enumerate(nums):
        if num in num_to_idx:
            num_to_idx[num].append(idx)
        else:
            num_to_idx[num] = [idx]

    max_length = 0
    max_start_idx = None
    max_subset_idx = []

    for num, idx_list in num_to_idx.items():
        for start_idx in idx_list:
            current_length = 1
            current_subset_idx = [start_idx]
            current_num = num

            while True:
                next_num = current_num + 1
                found = False
                while next_num - current_num <= gap and not found:
                    if next_num in num_to_idx:
                        for next_idx in num_to_idx[next_num]:
                            if next_idx > current_subset_idx[-1]:
                                current_length += 1
                                current_subset_idx.append(next_idx)
                                current_num = next_num
                                found = True
                                break  
                    next_num += 1

                if not found:
                    break

            if current_length > max_length:
                max_length = current_length
                max_start_idx = current_subset_idx[0]
                max_subset_idx = current_subset_idx

    max_subset = [nums[idx] for idx in max_subset_idx]
    return max_subset, max_subset_idx

def save_to_json(data, file_path, indent=2):
    
    def convert_numpy_to_list(data):
    
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {key: convert_numpy_to_list(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [convert_numpy_to_list(item) for item in data]
        else:
            return data

    converted_data = convert_numpy_to_list(data)
    with open(file_path, 'w') as json_file:
        json.dump(converted_data, json_file, indent=indent)

def split_list(lst, length):  
    if not isinstance(length, int) or length <= 0:  
        raise ValueError("Length must be a positive integer.")  
      
    return [lst[i:i+length] for i in range(0, len(lst), length)]


def vis_keypoint_in_img(img, j2d, point_color=(0, 128, 255), skeleton_colors=(0, 255, 0), str_text="name"):
    # j2d[[x,y],[x,y],...]
    if isinstance(img, str):
        img = cv2.imread(img)
    j2d = np.array(j2d)
    
    skeleton = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [1, 6], [3, 7], [4, 8],
                [5, 9], [6, 10], [7, 11], [8, 12], [9, 13], [10, 14]]
    skeleton = np.array(skeleton)

    if skeleton_colors:
        for pair in skeleton:
            partA = pair[0]
            partB = pair[1]
            if partA < len(j2d) and partB < len(j2d) and j2d[partA].any() and j2d[partB].any():
                cv2.line(img, (int(j2d[partA][0]), int(j2d[partA][1])),
                         (int(j2d[partB][0]), int(j2d[partB][1])), skeleton_colors, 2)
    else:
        skeleton_colors = {
            (0, 1): (255, 0, 0),  # blue
            (0, 2): (255, 0, 0),
            (0, 3): (0, 0, 255),
            (0, 4): (0, 255, 0),  # green
            (1, 5): (0, 0, 255),  # red
            (1, 6): (0, 255, 0),
            (3, 7): (0, 0, 255),
            (4, 8): (0, 255, 0),
            (5, 9): (0, 0, 255),
            (6, 10): (0, 255, 0),
            (7, 11): (0, 0, 255),
            (8, 12): (0, 255, 0),
            (9, 13): (0, 0, 255),
            (10, 14): (0, 255, 0)
        }

        for sk in skeleton:
            if sk[0] < len(j2d) and sk[1] < len(j2d) and j2d[sk[0]].any() and j2d[sk[1]].any():
                sk_tuple = tuple(sk)
                color = skeleton_colors.get(sk_tuple, (0, 255, 0))
                cv2.line(img, tuple(j2d[sk[0]].astype(int)), tuple(j2d[sk[1]].astype(int)), color, 2)

    for point in range(len(j2d)):
        if j2d[point].any():
            cv2.circle(img, (int(j2d[point][0]), int(j2d[point][1])), 2, point_color, -1)
            # cv2.putText(img, str(point), (int(j2d[point][0]+10), int(j2d[point][1])+10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    cv2.putText(img, str_text, (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 4)
    return img

def images_to_video(image_paths, output_video_path, frame_rate=30, video_size=(1920, 1080)):
    first_image = cv2.imread(os.path.join(image_paths, sorted(os.listdir(image_paths))[0]))
    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, video_size)
    for image_path in sorted(os.listdir(image_paths)):
        img = cv2.imread(os.path.join(image_paths, image_path))
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        video_writer.write(img)
    video_writer.release()

  
def concatenate_videos_horizontally(video1_path, video2_path, output_path):  
    cap1 = cv2.VideoCapture(video1_path)  
    cap2 = cv2.VideoCapture(video2_path)  
  
    fps = cap1.get(cv2.CAP_PROP_FPS)  
    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))  
    width2_orig = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height2_orig = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 输出视频编解码器  
  
    new_width2 = int((width2_orig / height2_orig) * height1)  
    new_height2 = height1  
  
    out = cv2.VideoWriter(output_path, fourcc, fps, (width1 + new_width2, height1))  
  
    while cap1.isOpened() and cap2.isOpened():  
        ret1, frame1 = cap1.read()  
        ret2, frame2 = cap2.read()  
  
        if ret1 and ret2:  
            frame2_resized = cv2.resize(frame2, (new_width2, new_height2))  
            concatenated_frame = np.hstack((frame1, frame2_resized))  
  
            out.write(concatenated_frame)  
        else:  
            break  
    print(f"{output_path} hase been saved")


  
def delete_directory(directory_path):  
    if os.path.exists(directory_path):  
        shutil.rmtree(directory_path)  
        print(f"Directory {directory_path} has been deleted.")  
    else:  
        print(f"Directory {directory_path} does not exist.")  