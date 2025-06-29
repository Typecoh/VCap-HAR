import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from MyCNN.enhanceNet import SimpleCNN


# 定义数据集路径和转换
def virtual_train(type):
    train_data_dir = "{}/pre_train_dataset/train".format(type)
    test_data_dir = "{}/pre_train_dataset/test".format(type)

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

    train_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_data_dir, transform=test_transform)

    batch_size = 5
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(2).to(device)  # 输出类别数设为2
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    best_model_path = "{}_pretrained_model_path.pth".format(type)
    best_accuracy = 0.0

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 9:
                print(f'epoch: {epoch + 1}, loss: {running_loss / 10:.3f}')
                running_loss = 0.0

        if epoch % 2 == 1:
            print('Testing...')
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            print(f'Accuracy: {100 * accuracy:.2f}%')

            if accuracy > best_accuracy:
                torch.save(model.state_dict(), best_model_path)
                best_accuracy = accuracy

    print('Finished Training')
    print(f'Best accuracy: {100 * best_accuracy:.2f}%, saved at {best_model_path}')


def fine_tune(type):
    train_data_dir = "{}/fine_tune_dataset/train".format(type)
    test_data_dir = "{}/fine_tune_dataset/test".format(type)

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_data_dir, transform=test_transform)

    batch_size = 5
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载前面训练好的两类模型
    pretrained_model_path = "{}_pretrained_model_path.pth".format(type)
    model = SimpleCNN(2)  # 初始类别数为2
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))

    # 修改全连接层以支持3类
    model.fc2 = nn.Linear(128, 3)  # 修改输出层为3类
    model = model.to(device)

    # 冻结卷积层
    for name, param in model.named_parameters():
        if "fc" not in name:  # 冻结所有非全连接层
            param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 100
    best_model_path = "{}_fine_tune_model_path.pth".format(type)
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 9:
                print(f'epoch: {epoch + 1}, loss: {running_loss / 10:.3f}')
                running_loss = 0.0

        if epoch % 2 == 1:
            print("Testing...")
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            print(f'Accuracy: {100 * accuracy:.2f}%')

            if accuracy > best_accuracy:
                torch.save(model.state_dict(), best_model_path)
                best_accuracy = accuracy

    print('Finished Fine-Tuning')
    print(f'Best accuracy: {100 * best_accuracy:.2f}%, saved at {best_model_path}')


if __name__ == '__main__':
    # 训练前两类
    for it in ["cap","res"]:
        # virtual_train(it)
        # 加载并微调剩余三类
        fine_tune(it)
