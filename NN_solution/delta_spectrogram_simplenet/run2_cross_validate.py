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
ready_data = np.load('../data/processed_data_250ms.npy')

shuffle_in_unison(ready_data, Tn)

print(ready_data.shape)
print(ready_data.dtype)
# Randomize
np.random.seed(seed=None)


Nparts = 10

t_best_err = np.zeros(Nparts)
t_best_loss = np.zeros(Nparts)
t_best_auc = np.zeros(Nparts)
t_best_fpr = []
t_best_tpr = []

for n_run in range(Nparts):

    x_train, t_train, x_dev, t_dev = gen_crossval_split(ready_data, Tn, n_run, Nparts, shuffle=False)
    fpr_tpr_set = False

    print('+train = %d' % np.sum(t_train))
    print('+t_dev = %d' % np.sum(t_dev))

    transform_train = transforms.Compose([
        # RandomCrop((192, 32), padding=(0, 0), pad_if_needed=False),
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
            roc_auc = metrics.auc(fpr, tpr)
            auc[i] = roc_auc

            cprint('g', '    Jdev = %f, err = %f\n' % (cost_dev[i], err_dev[i]))
            cprint('g', '    auc = %f\n' % (auc[i]))

            if auc[i] > best_auc:
                best_cost = cost_dev[i]
                best_auc = auc[i]
                net.save('models/theta_best.dat')

                ########## global stats #############
                t_best_auc[n_run] = auc[i]
                t_best_err[n_run] = err_dev[i]
                t_best_loss[n_run] = cost_dev[i]
                if not fpr_tpr_set:
                    t_best_fpr.append(fpr)
                    t_best_tpr.append(tpr)
                    fpr_tpr_set = True
                else:
                    t_best_fpr[n_run] = fpr
                    t_best_tpr[n_run] = tpr

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


plt.figure()
lw = 1.2
for i in range(Nparts):
    plt.plot(t_best_fpr[i], t_best_tpr[i],
         lw=lw, label='ROC curve (area = %0.3f)' % t_best_auc[i])

plt.plot([0, 1], [0, 1], color='C2', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(' Best Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('results/best_ROC_smallwindow.png')

print('% dev errors:', t_best_err)
print('% dev loss:', t_best_loss)