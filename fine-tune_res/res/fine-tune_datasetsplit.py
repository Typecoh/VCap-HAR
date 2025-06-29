import os
import shutil
from pathlib import Path
import random
# 定义源文件夹路径和目标文件夹路径
source_folder = "TrueDataset"
target_folder = "fine-tune_dataset"
# 创建目标文件夹结构
train_folder = os.path.join(target_folder, "train")
test_folder = os.path.join(target_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 遍历每个用户文件夹
for user in os.listdir(source_folder):
    user_path = os.path.join(source_folder, user)
    if os.path.isdir(user_path):  # 确保是文件夹
        # 遍历每个类别文件夹
        for category in os.listdir(user_path):
            category_path = os.path.join(user_path, category)
            if os.path.isdir(category_path):  # 确保是文件夹
                # 获取所有图片路径
                images = list(Path(category_path).glob("*.jpg"))  # 根据实际情况修改图片扩展名
                if len(images) < 10:
                    print(f"类别 {category} 图片不足 10 张，跳过...")
                    continue

                # 随机打乱图片顺序
                random.shuffle(images)

                # 分割为训练集和测试集
                train_images = images[:4]
                test_images = images[4:]

                # 创建对应类别文件夹
                train_category_folder = os.path.join(train_folder, category)
                test_category_folder = os.path.join(test_folder, category)
                os.makedirs(train_category_folder, exist_ok=True)
                os.makedirs(test_category_folder, exist_ok=True)

                # 移动训练图片
                for img_path in train_images:
                    shutil.copy(img_path, os.path.join(train_category_folder, img_path.name))

                # 移动测试图片
                for img_path in test_images:
                    shutil.copy(img_path, os.path.join(test_category_folder, img_path.name))

                print(f"用户 {user} 的类别 {category} 已处理完成：训练集 10 张，测试集 {len(test_images)} 张。")

print("数据集划分完成！")
