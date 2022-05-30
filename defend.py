import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from time import time
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from models import Net2
from utils import train, test, deepfool, fgsm_attack

train_dataset = MNIST('./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = MNIST('./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, 1000)
test_loader = DataLoader(test_dataset, 1000)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epsilons = np.array([1, 10, 20, 50]) / 255

net = Net2()
net.load_state_dict(torch.load('./network_weight.pth'))
net = net.to(device)
loss_fn = nn.CrossEntropyLoss()

'''
Generate adversarial examples using Untargeted FGSM and deepfool
'''
# generate Untargeted FGSM datasets
attack_acc = np.zeros(len(epsilons) + 1)
ut_fgsm_train_datasets = [MNIST('./data', train=True, transform=torchvision.transforms.ToTensor(), download=True) for _ in epsilons]
ut_fgsm_test_datasets = [MNIST('./data', train=False, transform=torchvision.transforms.ToTensor(), download=True) for _ in epsilons]
for i, epsilon in enumerate(epsilons):
    print('fgsm attacks with epsilon: {:.4f}'.format(epsilon))
    # generate the adversaril train_datasets
    total_correct = 0
    for k, (imgs, targets) in enumerate(train_loader):
        
        imgs = imgs.to(device)
        targets = targets.to(device)
        imgs.requires_grad = True
        output = net(imgs)
        # orig_targets = torch.argmax(output, dim=1)
        loss = loss_fn(output, targets)
        loss.backward()
        data_grad = imgs.grad
        pert_imgs, pert = fgsm_attack(imgs, data_grad, epsilon)

        pert_output = net(pert_imgs)
        pert_correct = sum(torch.argmax(pert_output, dim=1) == targets)
        total_correct += pert_correct
        pert_imgs = torch.floor(pert_imgs * 255)
        pert_imgs = torch.tensor(pert_imgs, dtype=torch.uint8)

        if k == 0:
            data = pert_imgs.squeeze().detach()
        else:
            data = torch.vstack((data, pert_imgs.squeeze().detach()))
    print('epsilon: {:.4f}, after attack train accuracy: {:.4f}'.format(epsilon, total_correct / len(train_dataset)))
    ut_fgsm_train_datasets[i].data = data

    # generate the adversarial test_datasets
    total_correct = 0
    for k, (imgs, target) in enumerate(test_loader):
        
        imgs = imgs.to(device)
        targets = targets.to(device)
        imgs.requires_grad = True
        output = net(imgs)
        # orig_targets = torch.argmax(output, dim=1)
        loss = loss_fn(output, targets)
        loss.backward()
        data_grad = imgs.grad
        pert_imgs, pert = fgsm_attack(imgs, data_grad, epsilon)

        pert_output = net(pert_imgs)
        pert_correct = sum(torch.argmax(pert_output, dim=1) == targets)
        total_correct += pert_correct
        pert_imgs = torch.floor(pert_imgs * 255)
        pert_imgs = torch.tensor(pert_imgs, dtype=torch.uint8)

        if k == 0:
            data = pert_imgs.squeeze().detach()
        else:
            data = torch.vstack((data, pert_imgs.squeeze().detach()))
    print('epsilon: {:.4f}, after attack test accuracy: {:.4f}'.format(epsilon, total_correct / len(test_dataset)))
    ut_fgsm_test_datasets[i].data = data
    attack_acc[i] = total_correct / len(test_dataset)

torch.save(ut_fgsm_train_datasets, './ut_fgsm_train_datasets.pt')
torch.save(ut_fgsm_test_datasets, './ut_fgsm_test_datasets.pt')

# generate deepfool datasets
df_train_loader = DataLoader(train_dataset, 1)  # deepfool is done on a single image
df_test_loader = DataLoader(test_dataset, 1)
net.eval()
# generate the adversaril train_dataset
start = time()
for i, (img, target) in enumerate(df_train_loader):
    img = img.to(device)
    target = target.to(device)
    output = net(img)
    orig_target = torch.argmax(output, dim=1)
    pert, pert_img, pert_target = deepfool(img.squeeze(0), net, overshoot=1e-5)
    pert_img = torch.floor(pert_img * 255)
    pert_img = torch.tensor(pert_img, dtype=torch.uint8)
    
    if i == 0:
        data = pert_img
    else:
        data = torch.vstack((data, pert_img))

    if (i + 1) % 1000 == 0:
        end = time()
        print('deepfool attack on train image {}, total time spent {:.4f}'.format(i + 1, end - start))

df_train_dataset = MNIST('./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
df_train_dataset.data = data
torch.save(df_train_dataset, './df_train_dataset.pt')

# generate the adversaril test_dataset
start = time()
for i, (img, target) in enumerate(df_test_loader):
    img = img.to(device)
    target = target.to(device)
    output = net(img)
    orig_target = torch.argmax(output, dim=1)
    pert, pert_img, pert_target = deepfool(img.squeeze(0), net, overshoot=1e-5)
    pert_img = torch.floor(pert_img * 255)
    pert_img = torch.tensor(pert_img, dtype=torch.uint8)
    
    if i == 0:
        data = pert_img.squeeze()
    else:
        data = torch.vstack((data, pert_img.squeeze()))

    if (i + 1) % 1000 == 0:
        end = time()
        print('deepfool attack on test image {}, total time spent {:.4f}'.format(i + 1, end - start))

df_test_dataset = MNIST('./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
df_test_dataset.data = data
torch.save(df_test_dataset, './df_test_dataset.pt')

'''
Use the augmented data to train the network and test the defence effect
'''
train_dataset = MNIST('./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = MNIST('./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

orig_data = train_dataset.data
orig_targets = train_dataset.targets

ut_fgsm_train_datasets = torch.load('./ut_fgsm_train_datasets.pt')
ut_fgsm_test_datasets = torch.load('./ut_fgsm_test_datasets.pt')
defend_acc = np.zeros((len(epsilons) + 1, len(epsilons) + 1))

for i, eps_train in enumerate(ut_fgsm_train_datasets):
    print('training data agumented by FGSM adversarial data with epsilon: {:.4f}'.format(epsilons[i]))
    train_dataset.data = torch.vstack((orig_data, eps_train.data))
    train_dataset.targets = torch.hstack((orig_targets, eps_train.targets))
    train_loader = DataLoader(train_dataset, 512)
    net_adv = Net2()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net_adv.parameters())
    train(net_adv, loss_fn, optimizer, train_loader)

    for j, eps_test in enumerate(ut_fgsm_test_datasets):
        print('test data: FGSM adversarial data with epsilon: {:.4f}'.format(epsilons[j]))
        test_dataset.data = eps_test.data
        test_dataset.targets = eps_test.targets
        test_loader = DataLoader(test_dataset, 1000)
        defend_acc[i][j] = test(net_adv, test_loader)
        print('train epsilon: {:.4f}, test epsilon: {:.4f}, defend accuracy: {:.4f}'.format(epsilons[i], epsilons[j], defend_acc[i][j]))

    print('test data: deepfool adversarial data')
    test_dataset = torch.load('./df_test_dataset.pt')
    test_loader = DataLoader(test_dataset)
    defend_acc[i][len(epsilons)] = test(net_adv, test_loader)
    print('train epsilon: {:.4f}, deepfool attack, defend accuracy: {:.4f}'.format(epsilons[i], defend_acc[i][len(epsilons)]))

df_train_dataset = torch.load('./df_train_dataset.pt')
train_dataset.data = torch.vstack((orig_data, df_train_dataset.data))
train_dataset.targets = torch.hstack((orig_targets, df_train_dataset.targets))
train_loader = DataLoader(train_dataset, 512)
net_adv_df = Net2()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net_adv_df.parameters())
train(net_adv_df, loss_fn, optimizer, train_loader)
for j, eps_test in enumerate(ut_fgsm_test_datasets):
    print('test data: FGSM adversarial data with epsilon: {:.4f}'.format(epsilons[j]))
    test_dataset.data = eps_test.data
    test_dataset.targets = eps_test.targets
    test_loader = DataLoader(test_dataset, 1000)
    defend_acc[len(epsilons)][j] = test(net_adv_df, test_loader)
    print('deepfool training, FGSM attack epsilon: {:.4f}, defend accuracy: {:.4f}'.format(epsilons[j], defend_acc[len(epsilons)][j]))
print('test data: deepfool adversarial data')
test_dataset = torch.load('./df_test_dataset.pt')
test_loader = DataLoader(test_dataset)
defend_acc[len(epsilons)][len(epsilons)] = test(net_adv_df, test_loader)
print('deepfool training, deepfool attack, defend accuracy: {:.4f}'.format(defend_acc[len(epsilons)][len(epsilons)]))

defend_acc = np.vstack((attack_acc, defend_acc))  
np.save('./ut_defend_acc', defend_acc)

'''
Show defend effect with untargeted training
'''
defend_acc = np.load('./ut_defend_acc.npy')
idx = np.linspace(0, 1, 6)
x_ticks = (
    'Non-adversarial',
    'FGSM {:.4f}'.format(epsilons[0]),
    'FGSM {:.4f}'.format(epsilons[1]),
    'FGSM {:.4f}'.format(epsilons[2]),
    'FGSM {:.4f}'.format(epsilons[3]),
    'Deepfool'
)

plt.plot(idx, defend_acc.T[0])
plt.title('Untargeted FGSM attack, epsilon: {:.4f}'.format(epsilons[0]))
plt.ylabel('accuracy after defend')
plt.xlabel('differnet adversarial methods for training')
plt.xticks(idx, x_ticks)
plt.show()

plt.plot(idx, defend_acc.T[1])
plt.title('Untargeted FGSM attack, epsilon: {:.4f}'.format(epsilons[1]))
plt.ylabel('accuracy after defend')
plt.xlabel('differnet adversarial methods for training')
plt.xticks(idx, x_ticks)
plt.show()

plt.plot(idx, defend_acc.T[2])
plt.title('Untargeted FGSM attack, epsilon: {:.4f}'.format(epsilons[2]))
plt.ylabel('accuracy after defend')
plt.xlabel('differnet adversarial methods for training')
plt.xticks(idx, x_ticks)
plt.show()

plt.plot(idx, defend_acc.T[3])
plt.title('Untargeted FGSM attack, epsilon: {:.4f}'.format(epsilons[3]))
plt.ylabel('accuracy after defend')
plt.xlabel('differnet adversarial methods for training')
plt.xticks(idx, x_ticks)
plt.show()

plt.plot(idx, defend_acc.T[4])
plt.title('Deepfool attack')
plt.ylabel('accuracy after defend')
plt.xlabel('differnet adversarial methods for training')
plt.xticks(idx, x_ticks)
plt.show()

idx = np.linspace(0, 1, 5)
x_ticks = (
    'FGSM {:.4f}'.format(epsilons[0]),
    'FGSM {:.4f}'.format(epsilons[1]),
    'FGSM {:.4f}'.format(epsilons[2]),
    'FGSM {:.4f}'.format(epsilons[3]),
    'Deepfool'
)

plt.plot(idx, defend_acc[0])
plt.title('No adversarial training')
plt.ylabel('accuracy')
plt.xlabel('differnet adversarial attacks')
plt.xticks(idx, x_ticks)
plt.show()

plt.plot(idx, defend_acc[1])
plt.title('Untargeted FGSM training, epsilon: {:.4f}'.format(epsilons[0]))
plt.ylabel('accuracy after defend')
plt.xlabel('differnet adversarial attacks')
plt.xticks(idx, x_ticks)
plt.show()

plt.plot(idx, defend_acc[2])
plt.title('Untargeted FGSM training, epsilon: {:.4f}'.format(epsilons[1]))
plt.ylabel('accuracy after defend')
plt.xlabel('differnet adversarial attacks')
plt.xticks(idx, x_ticks)
plt.show()

plt.plot(idx, defend_acc[3])
plt.title('Untargeted FGSM training, epsilon: {:.4f}'.format(epsilons[2]))
plt.ylabel('accuracy after defend')
plt.xlabel('differnet adversarial attacks')
plt.xticks(idx, x_ticks)
plt.show()

plt.plot(idx, defend_acc[4])
plt.title('Untargeted FGSM training, epsilon: {:.4f}'.format(epsilons[3]))
plt.ylabel('accuracy after defend')
plt.xlabel('differnet adversarial attacks')
plt.xticks(idx, x_ticks)
plt.show()

plt.plot(idx, defend_acc[5])
plt.title('Deepfool training')
plt.ylabel('accuracy after defend')
plt.xlabel('differnet adversarial attacks')
plt.xticks(idx, x_ticks)
plt.show()

'''
Generate adversarial examples using Targeted FGSM
'''
attack_acc = np.zeros(len(epsilons))
t_fgsm_train_datasets = [MNIST('./data', train=True, transform=torchvision.transforms.ToTensor(), download=True) for _ in epsilons]
t_fgsm_test_datasets = [MNIST('./data', train=False, transform=torchvision.transforms.ToTensor(), download=True) for _ in epsilons]
for i, epsilon in enumerate(epsilons):
    print('targeted fgsm attacks with epsilon: {:.4f}'.format(epsilon))
    # generate the adversaril train_datasets
    total_correct = 0
    for k, (imgs, targets) in enumerate(train_loader):
        
        imgs = imgs.to(device)
        targets = targets.to(device)
        imgs.requires_grad = True
        output = net(imgs)
        dest_targets = (targets + 1) % 10

        loss = loss_fn(output, dest_targets)
        loss.backward()
        data_grad = imgs.grad
        pert_imgs, pert = fgsm_attack(imgs, data_grad, epsilon, targeted=True)

        pert_output = net(pert_imgs)
        pert_correct = sum(torch.argmax(pert_output, dim=1) == targets)
        total_correct += pert_correct
        pert_imgs = torch.floor(pert_imgs * 255)
        pert_imgs = torch.tensor(pert_imgs, dtype=torch.uint8)

        if k == 0:
            data = pert_imgs.squeeze().detach()
        else:
            data = torch.vstack((data, pert_imgs.squeeze().detach()))
    print('epsilon: {:.4f}, after attack train accuracy: {:.4f}'.format(epsilon, total_correct / len(train_dataset)))
    t_fgsm_train_datasets[i].data = data

    # generate the adversarial test_datasets
    total_correct = 0
    for k, (imgs, targets) in enumerate(test_loader):
        
        imgs = imgs.to(device)
        targets = targets.to(device)
        imgs.requires_grad = True
        output = net(imgs)
        dest_targets = (targets + 1) % 10
        
        loss = loss_fn(output, dest_targets)
        loss.backward()
        data_grad = imgs.grad
        pert_imgs, pert = fgsm_attack(imgs, data_grad, epsilon, targeted=True)

        pert_output = net(pert_imgs)
        pert_correct = sum(torch.argmax(pert_output, dim=1) == targets)
        total_correct += pert_correct
        pert_imgs = torch.floor(pert_imgs * 255)
        pert_imgs = torch.tensor(pert_imgs, dtype=torch.uint8)

        if k == 0:
            data = pert_imgs.squeeze().detach()
        else:
            data = torch.vstack((data, pert_imgs.squeeze().detach()))
    print('epsilon: {:.4f}, after attack test accuracy: {:.4f}'.format(epsilon, total_correct / len(test_dataset)))
    t_fgsm_test_datasets[i].data = data
    attack_acc[i] = total_correct / len(test_dataset)

torch.save(t_fgsm_train_datasets, './t_fgsm_train_datasets.pt')
torch.save(t_fgsm_test_datasets, './t_fgsm_test_datasets.pt')

'''
Use the targeted attack samples to train the network and test the defend effect
'''
train_dataset = MNIST('./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = MNIST('./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)

orig_data = train_dataset.data
orig_targets = train_dataset.targets

t_fgsm_train_datasets = torch.load('./t_fgsm_train_datasets.pt')
t_fgsm_test_datasets = torch.load('./t_fgsm_test_datasets.pt')
defend_acc = np.zeros((len(epsilons), len(epsilons)))

for i, eps_train in enumerate(t_fgsm_train_datasets):
    print('training data agumented by FGSM adversarial data with epsilon: {:.4f}'.format(epsilons[i]))
    train_dataset.data = torch.vstack((orig_data, eps_train.data))
    train_dataset.targets = torch.hstack((orig_targets, eps_train.targets))
    train_loader = DataLoader(train_dataset, 512)
    net_adv = Net2()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net_adv.parameters())
    train(net_adv, loss_fn, optimizer, train_loader)

    for j, eps_test in enumerate(t_fgsm_test_datasets):
        print('test data: FGSM adversarial data with epsilon: {:.4f}'.format(epsilons[j]))
        test_dataset.data = eps_test.data
        test_dataset.targets = eps_test.targets
        test_loader = DataLoader(test_dataset, 1000)
        defend_acc[i][j] = test(net_adv, test_loader)
        print('train epsilon: {:.4f}, test epsilon: {:.4f}, defend accuracy: {:.4f}'.format(epsilons[i], epsilons[j], defend_acc[i][j]))

defend_acc = np.vstack((attack_acc, defend_acc))
np.save('./t_defend_acc', defend_acc)

'''
Show defend effect with targeted training
'''
defend_acc = np.load('./t_defend_acc.npy')
idx = np.linspace(0, 1, 5)
x_ticks = (
    'Non-adversarial',
    'FGSM {:.4f}'.format(epsilons[0]),
    'FGSM {:.4f}'.format(epsilons[1]),
    'FGSM {:.4f}'.format(epsilons[2]),
    'FGSM {:.4f}'.format(epsilons[3]),
)

plt.plot(idx, defend_acc.T[0])
plt.title('Targeted FGSM attack, epsilon: {:.4f}'.format(epsilons[0]))
plt.ylabel('accuracy after defend')
plt.xlabel('differnet adversarial methods for training')
plt.xticks(idx, x_ticks)
plt.show()

plt.plot(idx, defend_acc.T[1])
plt.title('Targeted FGSM attack, epsilon: {:.4f}'.format(epsilons[1]))
plt.ylabel('accuracy after defend')
plt.xlabel('differnet adversarial methods for training')
plt.xticks(idx, x_ticks)
plt.show()

plt.plot(idx, defend_acc.T[2])
plt.title('Targeted FGSM attack, epsilon: {:.4f}'.format(epsilons[2]))
plt.ylabel('accuracy after defend')
plt.xlabel('differnet adversarial methods for training')
plt.xticks(idx, x_ticks)
plt.show()

plt.plot(idx, defend_acc.T[3])
plt.title('Targeted FGSM attack, epsilon: {:.4f}'.format(epsilons[3]))
plt.ylabel('accuracy after defend')
plt.xlabel('differnet adversarial methods for training')
plt.xticks(idx, x_ticks)
plt.show()

idx = np.linspace(0, 1, 4)
x_ticks = (
    'FGSM {:.4f}'.format(epsilons[0]),
    'FGSM {:.4f}'.format(epsilons[1]),
    'FGSM {:.4f}'.format(epsilons[2]),
    'FGSM {:.4f}'.format(epsilons[3]),
)

plt.plot(idx, defend_acc[0])
plt.title('No adversarial training')
plt.ylabel('accuracy')
plt.xlabel('differnet adversarial attacks')
plt.xticks(idx, x_ticks)
plt.show()

plt.plot(idx, defend_acc[1])
plt.title('Targeted FGSM training, epsilon: {:.4f}'.format(epsilons[0]))
plt.ylabel('accuracy after defend')
plt.xlabel('differnet adversarial attacks')
plt.xticks(idx, x_ticks)
plt.show()

plt.plot(idx, defend_acc[2])
plt.title('Targeted FGSM training, epsilon: {:.4f}'.format(epsilons[1]))
plt.ylabel('accuracy after defend')
plt.xlabel('differnet adversarial attacks')
plt.xticks(idx, x_ticks)
plt.show()

plt.plot(idx, defend_acc[3])
plt.title('Targeted FGSM training, epsilon: {:.4f}'.format(epsilons[2]))
plt.ylabel('accuracy after defend')
plt.xlabel('differnet adversarial attacks')
plt.xticks(idx, x_ticks)
plt.show()

plt.plot(idx, defend_acc[4])
plt.title('Targeted FGSM training, epsilon: {:.4f}'.format(epsilons[3]))
plt.ylabel('accuracy after defend')
plt.xlabel('differnet adversarial attacks')
plt.xticks(idx, x_ticks)
plt.show()

'''
Build table
'''
untargeted = np.load('./ut_defend_acc.npy')
ut_rows = (
    'Non Adversarial Training',
    'FGSM {:.4f} Training'.format(epsilons[0]),
    'FGSM {:.4f} Training'.format(epsilons[1]),
    'FGSM {:.4f} Training'.format(epsilons[2]),
    'FGSM {:.4f} Training'.format(epsilons[3]),
    'Deepfool Training'
)
ut_cols = (
    'FGSM {:.4f} Test'.format(epsilons[0]),
    'FGSM {:.4f} Test'.format(epsilons[1]),
    'FGSM {:.4f} Test'.format(epsilons[2]),
    'FGSM {:.4f} Test'.format(epsilons[3]),
    'Deepfool Test'
)
ut_tab = plt.table(
    cellText=untargeted,
    rowLabels=ut_rows,
    colLabels=ut_cols,
    loc='center', cellLoc='center', rowLoc='center'
)
ut_tab.scale(1, 2)
plt.title('Untargeted Attack and Defend')
plt.axis('off')
plt.show()

targeted = np.load('./t_defend_acc.npy')
t_rows = (
    'Non Adversarial Training',
    'FGSM {:.4f} Training'.format(epsilons[0]),
    'FGSM {:.4f} Training'.format(epsilons[1]),
    'FGSM {:.4f} Training'.format(epsilons[2]),
    'FGSM {:.4f} Training'.format(epsilons[3]),
)
t_cols = (
    'FGSM {:.4f} Test'.format(epsilons[0]),
    'FGSM {:.4f} Test'.format(epsilons[1]),
    'FGSM {:.4f} Test'.format(epsilons[2]),
    'FGSM {:.4f} Test'.format(epsilons[3]),
)
t_tab = plt.table(
    cellText=untargeted,
    rowLabels=ut_rows,
    colLabels=ut_cols,
    loc='center', cellLoc='center', rowLoc='center',
)
t_tab.scale(1, 2)
plt.title('Targeted Attack and Defend')
plt.axis('off')
plt.show()