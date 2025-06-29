import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from MyCNN.enhanceNet import SimpleCNN
ACCS = []
sum = 0
# type = "automatic800"
for it in range(1,9):
    test_data_dir = "TrueDataset/user{}".format(it)
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
        # transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
        # transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
    ])

    # 加载自定义数据集
    test_dataset = torchvision.datasets.ImageFolder(root=test_data_dir, transform=test_transform)

    # 创建数据加载器
    batch_size = 5
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model_path = "train_train_five81.pth"
    # model_path = "11_23model/best2.pth"
    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(5).to(device)
    model.load_state_dict(torch.load(model_path))

    # 在测试集上评估模型
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            print("predicted: {} labels: {}".format(predicted,labels))
            correct += (predicted == labels).sum().item()
    sum = sum + (100 * correct / total)
    ACCS.append('Accuracy of the network on the test images: %.1f%%,correct = %d,total = %d' % (100 * correct / total, correct, total))
    # print('Accuracy of the network on the test images: %.1f%%' % (100 * correct / total))
for it in ACCS:
    print(it)
print(sum / 8)