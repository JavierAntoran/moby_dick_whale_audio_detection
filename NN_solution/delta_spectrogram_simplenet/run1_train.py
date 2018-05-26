from __future__ import print_function
from __future__ import division
import torch, time
import torch.utils.data
from torchvision import transforms
from sklearn import metrics
from src.datafeed import *
from model import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


mkdir('models')
mkdir('results')
# ------------------------------------------------------------------------------------------------------
# train config

batch_size = 64
nb_epochs = 35
log_interval = 1

# ------------------------------------------------------------------------------------------------------
# dataset
cprint('c', '\nData:')

# test train separation


# load data
Tn = np.load('../data/whale_trainlabels.npy')
ready_data = np.load('../data/processed_data.npy')

# debug
# Nim = 6
# print(Tn[Nim])
# plt.figure()
# plt.imshow(ready_data[Nim,:,:,0])
# plt.savefig('results/im0.png')
# plt.figure()
# plt.imshow(ready_data[Nim,:,:,1])
# plt.savefig('results/im1.png')
# plt.figure()
# plt.imshow(ready_data[Nim,:,:,2])
# plt.savefig('results/im2.png')
#

print(ready_data.shape)
print(ready_data.dtype)
# Randomize
np.random.seed(seed=None)

# shuffle_in_unison_scary(ready_data, Tn)

# train test split
cutoff = np.round(ready_data.shape[0] * 0.1).astype(int)
x_train = ready_data[cutoff:]
t_train = Tn[cutoff:]
print('+train = %d' % np.sum(t_train))

x_dev = ready_data[:cutoff]
t_dev = Tn[:cutoff]
print('+t_dev = %d' % np.sum(t_dev))

transform_train = transforms.Compose([
    RandomCrop((192, 32), padding=(0, 0), pad_if_needed=False),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    # RandomCrop((192, 32), padding=(0, 0), pad_if_needed=False),
    transforms.ToTensor()
])

trainset = Datafeed(x_train, t_train, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3)

testset = Datafeed(x_dev, t_dev, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=3)

## ---------------------------------------------------------------------------------------------------------------------
# net dims
cprint('c', '\nNetwork:')

use_cuda = True
lr = 1e-4
########################################################################################
net = Net(lr=lr, cuda=use_cuda)

epoch = 0

## ---------------------------------------------------------------------------------------------------------------------
# train
cprint('c', '\nTrain:')

print('  init cost variables:')
cost_train = np.zeros(nb_epochs)
cost_dev = np.zeros(nb_epochs)
err_train = np.zeros(nb_epochs)
err_dev = np.zeros(nb_epochs)
best_cost = np.inf

roc_probs = np.zeros(x_dev.shape[0])
roc_targets = np.zeros(x_dev.shape[0])
auc = np.zeros(nb_epochs)
best_auc = 0

nb_its_dev = 1

tic0 = time.time()
for i in range(epoch, nb_epochs):
    net.set_mode_train(True)

    tic = time.time()
    nb_samples = 0

    for x, y in trainloader:
        cost, err = net.fit(x, y)

        err_train[i] += err
        cost_train[i] += cost
        nb_samples += len(x)

    cost_train[i] /= nb_samples
    err_train[i] /= nb_samples

    toc = time.time()
    net.epoch = i
    # ---- print
    print("it %d/%d, Jtr = %f, err = %f, " % (i, nb_epochs, cost_train[i], err_train[i]), end="")
    cprint('r', '   time: %f seconds\n' % (toc - tic))

    # ---- dev
    if i % nb_its_dev == 0:
        net.set_mode_train(False)
        nb_samples = 0
        for j, (x, y) in enumerate(testloader):

            cost, err, probs = net.eval(x, y)
            roc_targets[nb_samples:nb_samples+len(x)] = y.numpy()
            roc_probs[nb_samples:nb_samples+len(x)] = probs.numpy()

            cost_dev[i] += cost
            err_dev[i] += err
            nb_samples += len(x)

        cost_dev[i] /= nb_samples
        err_dev[i] /= nb_samples

        fpr, tpr, threshold = metrics.roc_curve(roc_targets, roc_probs)
        print(tpr.shape)
        print(fpr.shape)
        roc_auc = metrics.auc(fpr, tpr)
        auc[i] = roc_auc

        cprint('g', '    Jdev = %f, err = %f\n' % (cost_dev[i], err_dev[i]))
        cprint('g', '    auc = %f\n' % (auc[i]))

        if auc[i] > best_auc:
            best_cost = cost_dev[i]
            best_auc = auc[i]
            net.save('models/theta_best.dat')
            # SAVE ROC CURVE
            plt.figure()
            lw = 1.2
            plt.plot(fpr, tpr, color='C0',
                     lw=lw, label='ROC curve (area = %0.3f)' % auc[i])
            plt.plot([0, 1], [0, 1], color='C2', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(' Best Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig('results/best_ROC_biggernet.png')


    net.save('models/theta_last.dat')
    save_obj([cost_train, cost_dev, err_train, err_dev, best_cost], 'models/cost.dat')

toc0 = time.time()
runtime_per_it = (toc0 - tic0) / float(nb_epochs)
cprint('r', '   average time: %f seconds\n' % runtime_per_it)

## ---------------------------------------------------------------------------------------------------------------------
# results
cprint('c', '\nRESULTS:')
nb_parameters = net.get_nb_parameters()
best_cost_dev = np.min(cost_dev)
best_cost_train = np.min(cost_train)
err_dev_min = err_dev[::nb_its_dev].min()
max_auc_dev = np.max(auc)

print('  cost_dev: %f (cost_train %f)' % (best_cost_dev, best_cost_train))
print('  err_dev: %f' % (err_dev_min))
print('  max_auc_dev: %f' % (max_auc_dev))
print('  nb_parameters: %d (%s)' % (nb_parameters, humansize(nb_parameters)))
print('  time_per_it: %fs\n' % (runtime_per_it))

with open('results/results.txt', 'w') as f:
    f.write(
        '%f %f %d %s %f\n' % (best_cost_dev, best_cost_train, nb_parameters, humansize(nb_parameters), runtime_per_it))

## ---------------------------------------------------------------------------------------------------------------------
# fig cost vs its

plt.figure()
fig, ax1 = plt.subplots()
ax1.plot(cost_train, 'r--')
ax1.plot(range(0, nb_epochs, nb_its_dev), cost_dev[::nb_its_dev], 'b-')
ax1.set_ylabel('J(CE) + CU cost')
plt.xlabel('epoch')
plt.grid(b=True, which='major', color='k', linestyle='-')
plt.grid(b=True, which='minor', color='k', linestyle='--')
plt.legend(['train cost', 'dev cost'])
plt.savefig('results/cost.png')

plt.figure()
fig2, ax2 = plt.subplots()
ax2.set_ylabel('% error')
ax2.semilogy(range(0, nb_epochs, nb_its_dev), 100 * err_dev[::nb_its_dev], 'b-')
ax2.semilogy(100 * err_train, 'r--')
plt.xlabel('epoch')
plt.grid(b=True, which='major', color='k', linestyle='-')
plt.grid(b=True, which='minor', color='k', linestyle='--')
ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.legend(['dev error', 'train error'])

# plt.show(block=False)
plt.savefig('results/err.png')

plt.figure()
fig2, ax2 = plt.subplots()
ax2.set_ylabel('AUC')
ax2.semilogy(range(0, nb_epochs, nb_its_dev), 100 * auc[::nb_its_dev], 'b-')
plt.xlabel('epoch')
plt.grid(b=True, which='major', color='k', linestyle='-')
plt.grid(b=True, which='minor', color='k', linestyle='--')
ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# plt.legend(['dev auc'])

plt.savefig('results/auc.png')