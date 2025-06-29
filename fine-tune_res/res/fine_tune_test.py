import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from MyCNN.enhanceNet import SimpleCNN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sum = 0
test_data_dir = "fine-tune_dataset/test"
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
])

# 加载自定义数据集
test_dataset = torchvision.datasets.ImageFolder(root=test_data_dir, transform=test_transform)

# 创建数据加载器
batch_size = 5
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model_path = "best_finetuned.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(5).to(device)
model.load_state_dict(torch.load(model_path))

# 在测试集上评估模型
model.eval()
correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())  # 收集预测值
        all_labels.extend(labels.cpu().numpy())  # 收集真实标签

# 计算准确率
accuracy = 100 * correct / total
print('Accuracy of the network on the test images: %.1f%%, correct = %d, total = %d' % (accuracy, correct, total))

# 计算并绘制混淆矩阵
cm = confusion_matrix(all_labels, all_preds)

# 使用 seaborn 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
