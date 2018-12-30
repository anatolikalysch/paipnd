import os
import time
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models


class NNTrainer:

    def __init__(self, args):

        assert args.data_dir is not None  # should never happen
        self.data_dir = args.data_dir
        self.train_dir = os.path.join(self.data_dir, 'train')
        self.val_dir = os.path.join(self.data_dir, 'valid')
        self.test_dir = os.path.join(self.data_dir, 'test')

        self.gpu = args.gpu

        self.epochs = args.epochs
        self.eps = args.epsilon_value
        self.lr = args.learning_rate
        self.arch = args.arch
        self.batch_size = args.batch
        self.hidden_units = args.hidden_units

        print('[*] hyperparams loaded')
        # dataset handling and image transformations
        self.__init_transforms__()
        print('[*] transformations loaded')

        self.__init_datasets__()
        print('[*] datasets loaded')

        self.__init_loaders__(args.batch)
        print('[*] loaders initiated')

        # model
        self.model = models.__dict__[self.arch](pretrained=True)
        self.__init_classifier__()
        print('[*] classifier loaded')

        # loss function and optimizer
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(),
                                    lr=args.learning_rate,
                                    eps=args.epsilon_value)

        if self.gpu:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
            print('[*] cuda init completed loaded')

        print('[*] Init finished.')

    def __init_transforms__(self):

        # normalization
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        # transformations
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize
            ]),
        }

    def __init_datasets__(self):
        # load datasets
        self.img_datasets = {
            'train': datasets.ImageFolder(self.train_dir, self.data_transforms['train']),
            'valid': datasets.ImageFolder(self.val_dir, self.data_transforms['valid']),
            'test': datasets.ImageFolder(self.test_dir, self.data_transforms['test']),
        }

    def __init_loaders__(self, batch_size):
        # dataloaders
        self.dataloaders = {
            'train': torch.utils.data.DataLoader(self.img_datasets['train'],
                                                 batch_size=batch_size,
                                                 shuffle=True),
            'valid': torch.utils.data.DataLoader(self.img_datasets['valid'],
                                                 batch_size=batch_size,
                                                 shuffle=True),
            'test': torch.utils.data.DataLoader(self.img_datasets['test'],
                                                batch_size=batch_size,
                                                shuffle=True),
        }

    def __init_classifier__(self):

        # Freeze parameters so we don't backpropagate through them
        for param in self.model.parameters():
            param.requires_grad = False

        default_units = {
            'vgg13': '25088, 6552, 1024, 512, 104',
            'vgg16': '25088, 6552, 1024, 512, 104',
            'vgg19': '25088, 6552, 1024, 512, 104'
        }

        if self.hidden_units is None:
            self.model.classifier = torch.nn.Sequential(
                OrderedDict([
                    ('dropout1', nn.Dropout()),
                    ('fc1', nn.Linear(25088, 6552)),
                    ('relu1', nn.ReLU()),
                    ('fc2', nn.Linear(6552, 1024)),
                    ('relu2', nn.ReLU()),
                    ('fc3', nn.Linear(1024, 512)),
                    ('relu3', nn.ReLU()),
                    ('fc4', nn.Linear(512, 104)),
                    ('output', nn.LogSoftmax(dim=1))
                ])
            )

        else:
            # out_features = len([i for i in os.listdir(self.train_dir)])

            hidden_units = self.hidden_units.split(',')
            hidden_number = len(hidden_units)
            seq = []
            seq.append(('dropout1', nn.Dropout()))
            for hu in range(hidden_number-1):
                seq.append(('fc{}'.format(hu+1), nn.Linear(int(hidden_units[hu]), int(hidden_units[hu+1]))))
                seq.append(('relu{}'.format(hu+1), nn.ReLU()))

            seq.pop(-1)
            seq.append(('output', nn.LogSoftmax(dim=1)))
            self.model.classifier = torch.nn.Sequential(
                OrderedDict(
                    seq
                )
            )

        # first
        #         first = [nn.Linear(25088, int(hidden_units[0])), relu, dropout]

        # middle
        # middle = []

    #         if hidden_number > 1:
    #             for i in range(hidden_number - 1):
    #                 middle.append(nn.Linear(int(hidden_units[i]), int(hidden_units[i + 1])))
    #                 middle.append(relu)
    #                 middle.append(dropout)

    #         else:
    #         middle.append(nn.Linear(2048, 1024))
    #         middle.append(relu)
    #         middle.append(dropout)
    # last
    #         last = [nn.Linear(int(hidden_units[-1]), out_features)]

    #         self.model.classifier = nn.Sequential(first, middle, last, output)

    def train(self):

        # GPU model configuration is handled during setup
        # if self.gpu:
        #     self.model = self.model.cuda()

        print('[*] Training started')

        start_time = time.time()
        steps = 0
        running_loss = 0
        print_every = 10

        for e in range(self.epochs):
            for img, labels in iter(self.dataloaders['train']):

                if self.gpu:
                    img, labels = img.cuda(), labels.cuda()

                steps += 1

                # wrap images and labels in Variables to calculate gradients
                inputs = Variable(img)
                targets = Variable(labels)
                self.optimizer.zero_grad()

                output = self.model.forward(inputs)
                loss = self.criterion(output, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.data.item()

                if steps % print_every == 0:
                    # model inference mode, dropout is off

                    accuracy = 0
                    val_loss = 0
                    for i, (img, labels) in enumerate(self.dataloaders['valid']):

                        if self.gpu:
                            img, labels = img.cuda(), labels.cuda()

                        with torch.no_grad():  # volatile is deprecated so we use no_grad to not save the history

                            inputs = Variable(img)  # , volatile=True)
                            labels = Variable(labels)  # , volatile=True)

                            output = self.model.forward(inputs)
                            val_loss += self.criterion(output, labels).item()

                            # calculating the accuracy: since models' output is log-softmax, take exponential
                            ps = torch.exp(output).data
                            # class with highest probability is our predicted class, compare with true label
                            equality = (labels.data == ps.max(1)[1])
                            # Accuracy is number of correct predictions
                            accuracy += equality.type_as(torch.FloatTensor()).mean()

                    print("Epoch:        {} of {}".format(e + 1, self.epochs),
                          "[T] Loss:     {:.2f}".format(running_loss / print_every),
                          "[V] Loss:     {:.2f}".format(val_loss / len(self.dataloaders['valid'])),
                          "[V] Accuracy: {:.2f} ({:.2f}%)".format(accuracy / len(self.dataloaders['valid']),
                                                                  100 * accuracy / len(self.dataloaders['valid'])))

                    running_loss = 0

                    # Make sure dropout is on for training
                    self.model.train()

        time_elapsed = time.time() - start_time

        print("\n[*] Training finished\n")
        print('[*] Time: {} sec\n'.format(time_elapsed))

    def test(self):

        self.model.eval()
        accuracy = 0
        test_loss = 0

        for i, (img, labels) in enumerate(self.dataloaders['test']):

            if self.gpu:
                img, labels = img.cuda(), labels.cuda()

            inputs = Variable(img, volatile=True)
            labels = Variable(labels, volatile=True)
            output = self.model.forward(inputs)
            test_loss += self.criterion(output, labels).data[0]

            ps = torch.exp(output).data
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        self.model.test_accuracy = accuracy.numpy() / len(self.dataloaders['test'])

        print("\nBatch: {} ".format(i + 1),
              "[-] Loss:     {:.2f}.. ".format(test_loss / len(self.dataloaders['test'])),
              "[+] Accuracy: {:.2f}\n".format(self.model.test_accuracy))

    def save(self):
        # CPU for loading compatability
        self.model = self.model.cpu()

        self.model.class_to_idx = self.dataloaders['train'].dataset.class_to_idx

        checkpoint = {
            'classifier': self.model.classifier,
            'state': self.model.state_dict(),
            'optimizer_dict': self.optimizer.state_dict(),
            'class_to_idx': self.model.class_to_idx,
            'hyperparams': {'arch': self.arch,
                            'epochs': self.epochs,
                            'lr': self.lr,
                            'eps': self.eps,
                            'hidden_units': self.hidden_units},
            'train_batch_size': self.dataloaders['train'].batch_size,
            'val_batch_size': self.dataloaders['valid'].batch_size,
            'test_batch_size': self.dataloaders['test'].batch_size,
            'accuracy': self.model.test_accuracy
        }

        checkpoint_path = os.path.relpath('{}_{}_udacity_ai.pth'.format(time.time(), self.arch))
        torch.save(checkpoint, checkpoint_path)
