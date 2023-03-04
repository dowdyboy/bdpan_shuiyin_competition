from torchvision.datasets import CIFAR10

import os
import random
import shutil


def generate_cifar10_image_folder(data_dir, out_dir):
    os.makedirs(os.path.join(out_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'test'), exist_ok=True)
    train_dataset = CIFAR10(data_dir, train=True, download=True, )
    test_dataset = CIFAR10(data_dir, train=False, download=True, )

    filename_idx = 0
    for i in range(len(train_dataset)):
        im, label = train_dataset[i]
        label_name = train_dataset.classes[label]
        label_dir = os.path.join(out_dir, 'train', label_name)
        os.makedirs(label_dir, exist_ok=True)
        im.save(os.path.join(label_dir, f'{filename_idx}.jpg'))
        filename_idx += 1

    filename_idx = 0
    for i in range(len(test_dataset)):
        im, label = test_dataset[i]
        label_name = test_dataset.classes[label]
        label_dir = os.path.join(out_dir, 'test', label_name)
        os.makedirs(label_dir, exist_ok=True)
        im.save(os.path.join(label_dir, f'{filename_idx}.jpg'))
        filename_idx += 1


def split_val_dataset_image_folder(data_train_dir, data_val_dir, rate=0.1):
    src_path_list = []
    label_set = set()
    for label_dir in os.listdir(data_train_dir):
        for filename in os.listdir(os.path.join(data_train_dir, label_dir)):
            src_path_list.append(os.path.join(data_train_dir, label_dir, filename))
            label_set.add(label_dir)
    for label_name in label_set:
        os.makedirs(os.path.join(data_val_dir, label_name), exist_ok=True)
    val_count = int(len(src_path_list) * rate)
    random.shuffle(src_path_list)
    for i in range(val_count):
        src_path = src_path_list[i]
        label_name = os.path.basename(os.path.dirname(src_path))
        file_name = os.path.basename(src_path)
        shutil.move(
            src_path,
            os.path.join(data_val_dir, label_name, file_name)
        )






