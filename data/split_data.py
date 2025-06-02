import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
import random

def split_dataset(source, train, val, test, train_split=0.7, val_split=0.2):
    os.makedirs(train, exist_ok=True)
    os.makedirs(val, exist_ok=True)
    os.makedirs(test, exist_ok=True)

    for class_name in os.listdir(source):
        src_class = os.path.join(source, class_name)
        if not os.path.isdir(src_class):
            continue

        files = os.listdir(src_class)
        random.shuffle(files)

        n = len(files)
        train_count = int(n * train_split)
        val_count = int(n * val_split)

        for i, file in enumerate(files):
            dst_dir = (
                train if i < train_count else
                val if i < train_count + val_count else
                test
            )
            os.makedirs(os.path.join(dst_dir, class_name), exist_ok=True)
            shutil.copy(os.path.join(src_class, file), os.path.join(dst_dir, class_name, file))

split_dataset('data/EarVN1.0/Images', 'data/EarVN1.0/train', 'data/EarVN1.0/val', 'data/EarVN1.0/test')