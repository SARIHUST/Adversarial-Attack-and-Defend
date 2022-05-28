import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(net, train_loader, test_loader, loss_fn, optimizer, epochs=10, write=False):
    if write:
        writer = SummaryWriter('./log')

    round = 0
    for i in range(epochs):
        print('================Epoch {}================'.format(i + 1))
        total_train_accurate, total_train_num, total_train_loss = 0, 0, 0
        net.to(device)
        net.train()
        for data in train_loader:
            # forward propagation
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net(inputs)

            # compute loss and other statistics
            loss = loss_fn(outputs, targets)
            y_predict = F.softmax(outputs, dim=1)
            accurate = sum(torch.argmax(y_predict, dim=1) == targets)
            total_train_accurate += accurate
            total_train_num += len(outputs)
            total_train_loss += loss

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # store train statistics
            if write and round % 100 == 0:
                writer.add_scalar('train loss on batch', loss, round)
                print('round {}, loss: {}'.format(round, loss))
                
            round += 1

        # compute test statistics of this epoch
        total_test_accurate, total_test_num, total_test_loss = 0, 0, 0
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, targets = data
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = net(inputs)

                loss = loss_fn(outputs, targets)
                y_predict = F.softmax(outputs, dim=1)
                accurate = sum(torch.argmax(y_predict, dim=1) == targets)
                total_test_accurate += accurate
                total_test_num += len(outputs)
                total_test_loss += loss
                
        # output statistics of this epoch
        train_acc = total_train_accurate / total_train_num
        test_acc = total_test_accurate / total_test_num
        print('train accuracy: {}'.format(train_acc))
        print('train loss: {}'.format(total_train_loss))
        print('test accuracy: {}'.format(test_acc))
        print('test loss: {}'.format(total_test_loss))

        if write:
            writer.add_scalar('train accuracy on each epoch', train_acc, i)
            writer.add_scalar('train loss on each epoch', total_train_loss, i)
            writer.add_scalar('test accuracy on each epoch', test_acc, i)
            writer.add_scalar('test loss on each epoch', total_test_loss, i)

    if write:
        writer.close()

def getKLayerFeatureMap(model_layers, k, x):
    '''
    x should have the size of 1 * C * H * W
    '''
    with torch.no_grad():
        for index, layer in enumerate(model_layers):
            x = layer(x)
            if index == k:
                return x

def showFeatureMap(feature_map):
    feature_map = feature_map.squeeze(0)
    channels = feature_map.shape[0]
    row_num = int(np.ceil(np.sqrt(channels)))
    for i in range(1, channels + 1):
        plt.subplot(row_num, row_num, i)
        plt.imshow(feature_map[i - 1], cmap='gray')
        plt.axis('off')
    plt.show()

def deepfool(img, net, k=10, overshoot=0.02, max_iter=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img.to(device)
    net.to(device)
    p_img = net(img.unsqueeze(0))
    idx = np.array(F.softmax(p_img.flatten(), dim=0).detach().numpy()).argsort()[::-1][0:k]

    label = idx[0]
    img_pert = img.clone().detach()
    w, r = np.zeros(img.shape), np.zeros(img.shape)
    
    x = img_pert.unsqueeze(0).float()
    x.requires_grad = True
    px = net(x)[0]
    lx = label
    it = 0
    while lx == label and it < max_iter:
        pert = np.inf
        px[label].backward(retain_graph=True)
        orig_grad = x.grad.data.numpy().copy()

        # find the minimum perturbation to change to another label
        for i in range(1, k):
            x.grad = None
            px[idx[i]].backward(retain_graph=True)
            i_grad = x.grad.data.numpy().copy()

            w_i = i_grad - orig_grad
            f_i = px[idx[i]] - px[idx[0]]
            pert_i = abs(f_i) / np.linalg.norm(w_i.flatten())

            if pert_i < pert:
                pert = pert_i
                w = w_i
        # compute the total perturbation at this round
        pert = pert.detach().numpy()
        rx = (pert + 1e-7) * w / np.linalg.norm(w)
        r += rx.squeeze(0)
        # compute the new image
        img_pert = (img + (1 + overshoot) * torch.from_numpy(r)).to(device)
        x = img_pert.unsqueeze(0).float()
        x.requires_grad = True
        px = net(x)[0]
        # print(px)
        lx = torch.argmax(px)

        it += 1

    r = (1 + overshoot) * r
    return r, img_pert, lx


def fgsm_attack(img, data_grad, epsilon):
    sign_data_grad = data_grad.sign()
    pert = epsilon * sign_data_grad
    pert_img = img + pert
    pert_img = torch.clamp(pert_img, 0, 1)
    return pert_img, pert