import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from MyCNN.enhanceNet import SimpleCNN


# 定义数据集路径和转换
def virtual_train():
    train_data_dir = "VirtualDataset"
    test_data_dir = "TrueDataset/user5"
    # test_data_dir = "../DataSets/datasets4/test"
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
    train_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_data_dir, transform=test_transform)

    # 创建数据加载器
    batch_size = 5
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    best_model_path = "virtual_best.pth"
    best_accuracy = 0.0

    # 训练模型
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            print(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print('epoch: %d, loss: %.3f' %
                      (epoch + 1, running_loss / 10))
                running_loss = 0.0
        if epoch % 2 == 1:
            print('Testing')
            model.eval()  # 设置模型为评估模式
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
            print('Accuracy of the network on the test images: %d %%' % (100 * accuracy))

            # 如果当前模型精度更高，则保存当前模型
            if accuracy > best_accuracy:
                torch.save(model.state_dict(), best_model_path)
                best_accuracy = accuracy

    print('Finished Training')
    print(f"Best accuracy: {100 * best_accuracy:.2f}%, saved at {best_model_path}")

'''
从4000（5*800）张真实数据中取出来400（5*80）张做fine-tune，在剩余的（5*420）张测试，看精度
'''
# 定义数据集路径和转换

def fine_tune():
    train_data_dir = "fine-tune_dataset/train"
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
    train_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_data_dir, transform=test_transform)

    # 创建数据加载器
    batch_size = 5
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(5).to(device)

    # 加载预训练模型参数
    # pretrained_model_path = "virtual_best.pth"
    pretrained_model_path = "400_best1.pth"
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))

    # 冻结卷积层，只微调全连接层
    for name, param in model.named_parameters():
        if "fc" not in name:  # 只更新全连接层
            param.requires_grad = False

    # 优化器仅更新 requires_grad=True 的参数
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001)
    criterion = nn.CrossEntropyLoss()

    # 微调模型
    num_epochs = 50
    best_model_path = "best_finetuned.pth"
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
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

        # 在测试集上评估
        if epoch % 2 == 1:
            print("Testing...")
            model.eval()  # 设置模型为评估模式
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
            print(f'Accuracy of the network on the test images: {100 * accuracy:.2f}%')

            # 如果当前模型精度更高，则保存
            if accuracy > best_accuracy:
                torch.save(model.state_dict(), best_model_path)
                best_accuracy = accuracy

    print('Finished Fine-Tuning')
    print(f"Best accuracy after fine-tuning: {100 * best_accuracy:.2f}%, saved at {best_model_path}")

if __name__ == '__main__':
    # 先使用virtual数据训练
    # virtual_train()
    # 调用 fine_tune
    fine_tune()

