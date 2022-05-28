import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from time import time
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from models import Net2
from utils import deepfool, fgsm_attack

train_dataset = MNIST('./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = MNIST('./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, 128)
test_loader = DataLoader(test_dataset, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = Net2()
net.load_state_dict(torch.load('./network_weight.pth'))
net = net.to(device)
loss_fn = nn.CrossEntropyLoss()

'''
First part: non-targeted attack
    use 2 types of attack - 
        1. Fast Gradient Signed Method
        2. Deepfool Method
    use 2 metrics to assess the performance -
        1. attack success rate ASR = No(attack success) / No(predict correct)
        2. rou = E(sum(|perturbation| / |img|)) -> indicates how different the adversial
    example is to the original image
'''
# FGSM attack
epsilons = np.array([0, 1, 10, 20, 50]) / 255
adv_examples = [[] for _ in epsilons]
asr_values = np.zeros_like(epsilons)
rou_value = np.zeros_like(epsilons)
show_num = 5
net.eval()
for i, epsilon in enumerate(epsilons):
    print('fgsm attacks with epsilon: {:.3}'.format(epsilon))
    correct, rou, error = 0, 0, 0
    start = time()
    for k, (img, target) in enumerate(test_loader):
        
        img = img.to(device)
        target = target.to(device)
        img.requires_grad = True
        output = net(img)
        orig_target = torch.argmax(output, dim=1)
        if orig_target != target:
            error += 1
            continue

        loss = loss_fn(output, target)
        loss.backward()
        data_grad = img.grad
        pert_img, pert = fgsm_attack(img, data_grad, epsilon)

        pert_output = net(pert_img)
        pert_target = torch.argmax(pert_output, dim=1)

        if pert_target == target:
            correct += 1
        else:
            rou += pert.norm() / img.norm()
        if len(adv_examples[i]) < show_num and (pert_target != target or i == 0):
            img = img.squeeze().detach().numpy()
            pert_img = pert_img.squeeze().detach().numpy()
            adv_examples[i].append((orig_target.item(), pert_target.item(), img, pert_img))
        
        if (k + 1) % 2000 == 0:
            end = time()
            print('fgsm attack on test image {}, total time spent {:.3f}'.format(k + 1, end - start))
    
    total_num = len(test_dataset)                       # No.(imgs)
    attack_num = total_num - error                      # No.(attacked imgs)
    attack_success_num = total_num - error - correct    # No.(attacked succeeded imgs)
    acc_rate = correct / total_num
    if i == 0:
        rou_value[i] = np.inf
        asr_values[i] = 0.0
        print('original accuracy: {}'.format(acc_rate))
    else:
        rou_value[i] = rou / attack_success_num
        asr_values[i] = attack_success_num / attack_num
        print('epsilon: {:.3}, accuracy: {:.3}, attack success rate: {:.3}, rou value: {:.3}'.format(epsilon, acc_rate, asr_values[i], rou_value[i]))

np.save('./fgsm_adv_examples', np.array(adv_examples))

adv_examples = np.load('./fgsm_adv_examples.npy', allow_pickle=True)
row, col, idx = len(epsilons), show_num, 0
plt.figure(figsize=(10, 10))
for i in range(row):
    for j in range(show_num):
        orig_target, pert_target, img, pert_img = adv_examples[i][j]
        idx += 1
        plt.subplot(row, col * 2, idx)
        if j == 0:
            plt.ylabel('Eps: {:.3}'.format(epsilons[i]))
        plt.title('Real: {}'.format(orig_target))
        plt.imshow(img, cmap='gray')
        idx += 1
        if i != 0:
            plt.subplot(row, col * 2, idx)
            plt.title('Adv: {}'.format(pert_target))
            plt.imshow(pert_img, cmap='gray')
plt.tight_layout()
plt.show()

# deepfool attack
show_num = 15
df_examples = []
rou, correct, error = 0, 0, 0
net.eval()
start = time()
for i, (img, target) in enumerate(test_loader):
    img = img.to(device)
    target = target.to(device)
    output = net(img)
    orig_target = torch.argmax(output, dim=1)
    if orig_target != target:
        error += 1
        continue
    pert, pert_img, pert_target = deepfool(img.squeeze(0), net, overshoot=1e-5)
    if pert_target == orig_target:
        correct += 1
    else:
        rou += np.linalg.norm(pert) / img.norm()
        if len(df_examples) < show_num:
            img = img.squeeze().detach().numpy()
            pert_img = pert_img.squeeze().detach().numpy()
            pert = pert.squeeze()
            df_examples.append((target.item(), pert_target.item(), img, pert_img, pert))

    if (i + 1) % 1000 == 0:
        end = time()
        print('deepfool attack on test image {}, total time spent {:.3f}'.format(i + 1, end - start))

rou /= len(train_dataset) - error - correct
orig_accuracy = (len(test_dataset) - error) / len(test_dataset)
df_asr = (len(train_dataset) - error - correct) / (len(train_dataset) - error)
print('original accuracy: {:.3}, deepfool attack success rate: {:.3}, rou value: {:.3}'.format(orig_accuracy, df_asr, rou))

np.save('./df_examples', np.array(df_examples))

df_examples = np.load('./df_examples.npy', allow_pickle=True)
plt.figure(figsize=(10, 10))
idx = 0
for i in range(5):
    for j in range(3):
        orig_target, pert_target, img, pert_img, pert = df_examples[i * 3 + j]
        idx += 1
        plt.subplot(5, 9, idx)
        plt.title('Real: {}'.format(orig_target))
        plt.imshow(img, cmap='gray')
        idx += 1
        plt.subplot(5, 9, idx)
        plt.title('Pert')
        plt.imshow(pert, cmap='gray')
        idx += 1
        plt.subplot(5, 9, idx)
        plt.title('Adv: {}'.format(pert_target))
        plt.imshow(pert_img, cmap='gray')

plt.tight_layout()
plt.show()