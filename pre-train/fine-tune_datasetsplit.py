import os
import shutil
from pathlib import Path
import random
# 定义源文件夹路径和目标文件夹路径

def pre_train_pre_train_dataset():
    # train ==> 两类 virtual data
    # test  ==> 两类 true data
    rootdir = ["res"]

    for type in rootdir:
        source_folder_train = f"../dataset/{type}/virtual"
        source_folder_test = f"../dataset/{type}/True"

        target_folder_train = f"{type}/pre_train_dataset/train"
        target_folder_test = f"{type}/pre_train_dataset/test"

        # 创建目标文件夹，如果不存在
        os.makedirs(target_folder_train, exist_ok=True)
        os.makedirs(os.path.join(target_folder_test, "01"), exist_ok=True)
        os.makedirs(os.path.join(target_folder_test, "02"), exist_ok=True)

        # 处理 train 数据集（virtual）
        subfolders = sorted(os.listdir(source_folder_train))  # 排序以确保01、02优先
        for subfolder in subfolders[:2]:  # 只取前两个
            source_path = os.path.join(source_folder_train, subfolder)
            target_path = os.path.join(target_folder_train, subfolder)

            if os.path.isdir(source_path):  # 确保是文件夹
                print(f"正在复制 {source_path} 到 {target_path}")
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)  # 复制文件夹及其内容
            else:
                print(f"{source_path} 不是一个有效的文件夹，跳过。")

        # 处理 test 数据集（True）
        users = sorted(os.listdir(source_folder_test))  # 获取 user1, user2, ..., user8

        users = [users[0]]

        for user in users:
            user_path = os.path.join(source_folder_test, user)

            if os.path.isdir(user_path):  # 确保是文件夹
                # 只处理 01 和 02 文件夹
                for folder in ["01", "02"]:
                    source_path = os.path.join(user_path, folder)
                    target_path = os.path.join(target_folder_test, folder)

                    if os.path.isdir(source_path):  # 确保是文件夹
                        print(f"正在复制 {source_path} 到 {target_path}")
                        for file_name in os.listdir(source_path):
                            file_source = os.path.join(source_path, file_name)
                            file_target = os.path.join(target_path, file_name)

                            if os.path.isfile(file_source):  # 确保是文件
                                shutil.copy2(file_source, file_target)  # 复制文件
                            else:
                                print(f"{file_source} 不是有效的文件，跳过。")
                    else:
                        print(f"{source_path} 不是一个有效的文件夹，跳过。")

    print("文件复制完成！")



def pre_train_fine_tune_dataset():
    # train ==> 三类 part true data (随机选取10张)
    # test  ==> 三类 other true data (剩余数据)
    rootdir = ["res"]

    for type in rootdir:
        source_folder = f"../dataset/{type}/True"
        target_folder_train = f"{type}/fine_tune_dataset/train"
        target_folder_test = f"{type}/fine_tune_dataset/test"

        # 创建目标文件夹
        os.makedirs(target_folder_train, exist_ok=True)
        os.makedirs(target_folder_test, exist_ok=True)

        # 遍历用户文件夹（如 user1, user2, ...）
        users = sorted(os.listdir(source_folder))  # 假设子文件夹按 user 排列
        for user in users:
            user_path = os.path.join(source_folder, user)

            if os.path.isdir(user_path):  # 确保是文件夹
                # 处理后三类 (03, 04, 05)
                for category in ["03", "04", "05"]:
                    category_path = os.path.join(user_path, category)

                    if os.path.isdir(category_path):  # 确保是有效文件夹
                        # 获取该类文件夹中的所有图片
                        image_files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
                        if len(image_files) == 0:
                            print(f"{category_path} 中没有图片，跳过。")
                            continue

                        # 随机选取10张图片作为 train，其余作为 test
                        random.shuffle(image_files)
                        train_files = image_files[:10]
                        test_files = image_files[10:]

                        # 创建目标文件夹
                        train_target_folder = os.path.join(target_folder_train, category)
                        test_target_folder = os.path.join(target_folder_test, category)
                        os.makedirs(train_target_folder, exist_ok=True)
                        os.makedirs(test_target_folder, exist_ok=True)

                        # 复制 train 文件
                        for file in train_files:
                            source_file = os.path.join(category_path, file)
                            target_file = os.path.join(train_target_folder, f"{user}_{file}")  # 防止文件名冲突
                            shutil.copy2(source_file, target_file)

                        # 复制 test 文件
                        for file in test_files:
                            source_file = os.path.join(category_path, file)
                            target_file = os.path.join(test_target_folder, f"{user}_{file}")  # 防止文件名冲突
                            shutil.copy2(source_file, target_file)

                        print(f"{user}/{category} 数据已完成：train {len(train_files)} 张, test {len(test_files)} 张")
                    else:
                        print(f"{category_path} 不是有效的文件夹，跳过。")

    print("文件处理完成！")


if __name__ == '__main__':
    # pre_train_pre_train_dataset()
    pre_train_fine_tune_dataset()