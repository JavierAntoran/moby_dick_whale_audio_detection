from torch.backends import cudnn
from src.utils import *
from simplenet import *

#
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#
class BaseNet(object):
    def __init__(self):
        cprint('c', '\nNet:')

    def get_nb_parameters(self):
        return np.sum(p.numel() for p in self.model.parameters())

    def set_mode_train(self, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def update_lr(self, epoch, gamma=0.99):
        self.epoch += 1
        if self.schedule is not None:
            if len(self.schedule) == 0 or epoch in self.schedule:
                self.lr *= gamma
                print('learning rate: %f  (%d)\n' % self.lr, epoch)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

    def save(self, filename):
        cprint('c', 'Writting %s\n' % filename)
        torch.save({
            'epoch': self.epoch,
            'lr': self.lr,
            'model': self.model,
            'optimizer': self.optimizer}, filename)

    def load(self, filename):
        cprint('c', 'Reading %s\n' % filename)
        state_dict = torch.load(filename)
        self.epoch = state_dict['epoch']
        self.lr = state_dict['lr']
        self.model = state_dict['model']
        self.optimizer = state_dict['optimizer']
        print('  restoring epoch: %d, lr: %f' % (self.epoch, self.lr))
        return self.epoch


class Net(BaseNet):
    eps = 1e-6

    def __init__(self, lr=1e-3, channels_in=3, adapt_shape=False, cuda=True):
        super(Net, self).__init__()
        cprint('y', ' simplenet ')
        self.lr = lr
        self.schedule = None  # [] #[50,200,400,600]
        self.cuda = cuda
        self.channels_in = channels_in
        self.create_net()
        self.create_opt()
        self.epoch = 0
        self.adapt_shape = adapt_shape

        self.test=False

    def create_net(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        self.model = simplenet(classes=2, channels_in=self.channels_in)
        self.downsample_model = channel_downsampler(channels_in=120, channels_out=64)
        if self.cuda:
            self.model.cuda()
            self.downsample_model.cuda()
            cudnn.benchmark = True

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
                                          weight_decay=0)

    #         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
    # self.sched = torch.optim.lr_scheduler.LambdaLR(self.optimizer, flg.decay)

    def run_adapt_shape(self, x):

        batch_size  = x.shape[0]
        x = x.squeeze(dim=1)
        normal_shape = x[:, :, :96].contiguous().view(batch_size, 160, 32, 3).permute(0,3,1,2)
        multires_features = x[:, :, 96:].permute(0,2,1)        # batch_size, 120, 160

        if not self.test:
            self.test = True
            plt.figure()
            plt.imshow(normal_shape[0, 0].data.cpu().numpy().T, cmap='jet', aspect='auto')
            plt.gca().invert_yaxis()
            plt.savefig('test0.png')
            plt.figure()
            plt.imshow(normal_shape[0, 1].data.cpu().numpy().T, cmap='jet', aspect='auto')
            plt.gca().invert_yaxis()
            plt.savefig('test1.png')
            plt.figure()
            plt.imshow(normal_shape[0, 2].data.cpu().numpy().T, cmap='jet', aspect='auto')
            plt.gca().invert_yaxis()
            plt.savefig('test2.png')
            plt.figure()
            plt.imshow(multires_features[0].data.cpu().numpy(), cmap='jet', aspect='auto')
            plt.gca().invert_yaxis()
            plt.savefig('test3.png')
            plt.figure()
            # plt.imshow(normal_shape[0, 4].data.cpu().numpy().T, cmap='jet', aspect='auto')
            # plt.gca().invert_yaxis()
            # plt.savefig('test4.png')

        multires_features = self.downsample_model(multires_features)

        multires_features = multires_features.permute(0, 2, 1).contiguous().view(batch_size, 160, 32, 2).permute(0, 3, 1, 2)

        normal_shape = torch.cat((normal_shape, multires_features), dim=1)

        return normal_shape

    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        self.optimizer.zero_grad()
        if self.adapt_shape:
            x = self.run_adapt_shape(x)

        out = self.model(x)
        loss = F.cross_entropy(out, y, size_average=True)

        loss.backward()
        self.optimizer.step()

        # out: (batch_size, out_channels, out_caps_dims)
        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data[0], err

    def eval(self, x, y, train=False):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        if self.adapt_shape:
            x = self.run_adapt_shape(x)

        out = self.model(x)

        loss = F.cross_entropy(out, y, size_average=True)

        probs = F.softmax(out, dim=1)[:, 1].data.cpu()

        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data[0], err, probs
