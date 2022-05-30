import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from time import time
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from models import Net2
from utils import train, deepfool, fgsm_attack, test

train_dataset = MNIST('./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_dataset = MNIST('./data', train=False, transform=torchvision.transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, 128)
test_loader = DataLoader(test_dataset, 1)   
# change 1 image each batch to get better adversarial examples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = Net2()
net.load_state_dict(torch.load('./network_weight.pth'))
net = net.to(device)
loss_fn = nn.CrossEntropyLoss()
final_acc = test(net, test_loader)
print('original accuracy: {:.4f}'.format(final_acc))

'''
First part: Untargeted Attack
    use 2 types of attack - 
        1. Fast Gradient Signed Method
        2. Deepfool Method
    use 2 metrics to assess the performance -
        1. attack success rate ASR = No(attack success) / No(predict correct)
        2. rou = E(sum(|perturbation| / |img|)) -> indicates how different the adversarial
    example is to the original image
'''
# FGSM attack
epsilons = np.array([1, 10, 20, 50]) / 255
fgsm_examples = [[] for _ in epsilons]
asr_values = np.zeros_like(epsilons)
rou_value = np.zeros_like(epsilons)
net.eval()
for i, epsilon in enumerate(epsilons):
    examples_found = np.zeros(10)
    print('fgsm attacks with epsilon: {:.4f}'.format(epsilon))
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
            if examples_found[orig_target] == 0:
                examples_found[orig_target] = 1
                img = img.squeeze().detach().numpy()
                pert_img = pert_img.squeeze().detach().numpy()
                fgsm_examples[i].append((orig_target.item(), pert_target.item(), img, pert_img))
        
        if (k + 1) % 2000 == 0:
            end = time()
            print('fgsm attack on test image {}, total time spent {:.4f}'.format(k + 1, end - start))
    
    total_num = len(test_dataset)                       # No.(imgs)
    attack_num = total_num - error                      # No.(attacked imgs)
    attack_success_num = total_num - error - correct    # No.(attacked succeeded imgs)
    acc_rate = correct / total_num

    rou_value[i] = rou / attack_success_num
    asr_values[i] = attack_success_num / attack_num
    print('epsilon: {:.4f}, accuracy: {:.4f}, attack success rate: {:.4f}, rou value: {:.4f}'.format(epsilon, acc_rate, asr_values[i], rou_value[i]))

row, col= len(epsilons), 20
plt.figure(figsize=(10, 10))
for i in range(row):
    idx = i * col
    for j in range(len(fgsm_examples[i])):
        orig_target, pert_target, img, pert_img = fgsm_examples[i][j]
        idx += 1
        plt.subplot(row, col, idx)
        if j == 0:
            plt.ylabel('Eps: {:.4f}'.format(epsilons[i]))
        plt.title('Real: {}'.format(orig_target))
        plt.imshow(img, cmap='gray')
        idx += 1
        plt.subplot(row, col, idx)
        plt.title('Adv: {}'.format(pert_target))
        plt.imshow(pert_img, cmap='gray')
plt.tight_layout()
plt.show()

# deepfool attack
df_examples = [[] for _ in range(10)]
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
        if len(df_examples[orig_target]) < 2:
            img = img.squeeze().detach().numpy()
            pert_img = pert_img.squeeze().detach().numpy()
            pert = pert.squeeze()
            df_examples[orig_target].append((target.item(), pert_target.item(), img, pert_img, pert))

    if (i + 1) % 1000 == 0:
        end = time()
        print('deepfool attack on test image {}, total time spent {:.4f}'.format(i + 1, end - start))

rou /= len(train_dataset) - error - correct
orig_accuracy = (len(test_dataset) - error) / len(test_dataset)
df_asr = (len(train_dataset) - error - correct) / (len(train_dataset) - error)
print('original accuracy: {:.4f}, deepfool attack success rate: {:.4f}, rou value: {:.4f}'.format(orig_accuracy, df_asr, rou))

torch.save(df_examples, './df_ex.pt')

df_examples = torch.load('./df_ex.pt')

plt.figure(figsize=(20, 20))
idx = 0
for i in range(4):
    for j in range(5):
        x, y = (i % 2) * 5 + j, i // 2
        orig_target, pert_target, img, pert_img, pert = df_examples[x][y]
        idx += 1
        plt.subplot(4, 15, idx)
        plt.title('Real: {}'.format(orig_target))
        plt.imshow(img, cmap='gray')
        idx += 1
        plt.subplot(4, 15, idx)
        plt.title('Pert')
        plt.imshow(pert, cmap='gray')
        idx += 1
        plt.subplot(4, 15, idx)
        plt.title('Adv: {}'.format(pert_target))
        plt.imshow(pert_img, cmap='gray')

plt.tight_layout()
plt.show()

'''
Second part: targeted FGSM attack
    try to change the label of the testing data, 0 -> 1, 1 -> 2, ..., 8 -> 9, 9 -> 0
'''

