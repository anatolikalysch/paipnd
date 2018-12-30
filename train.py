#! /usr/bin/env python3
# coding=utf-8

import argparse

from trainer import NNTrainer

MODEL_ARCH = ['vgg13', 'vgg16', 'vgg19']


def main():
    args = get_input_args()

    nn_t = NNTrainer(args)

    nn_t.train()

    nn_t.test()

    nn_t.save()


def get_input_args():

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('data_dir', metavar='DIR', help='path to dataset')

    parser.add_argument('--epochs', dest='epochs', default=4, type=int, help='number of epochs')
    parser.add_argument('--batch-size', dest='batch', default=64, type=int, help='batch size')

    parser.add_argument('--learning_rate', dest='learning_rate', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--hidden_units', dest='hidden_units', default=None, type=str, help='define custom hidden '
                                                                                            'layers by providing a '
                                                                                            'comma separated list: '
                                                                                            'i_0,i_1, ... ,'
                                                                                            'i_n of which values to '
                                                                                            'use')
    parser.add_argument('--epsilon_value', dest='epsilon_value', default=1e-6, type=float, help='epsilon value')

    parser.add_argument('--arch', dest='arch', default='vgg19', choices=MODEL_ARCH, help='choose model architecture: VGG < 13 | 16 | 19 >')
    parser.add_argument('--gpu', dest='gpu', default=False, action='store_true', help='use gpu')

    return parser.parse_args()


if __name__ == '__main__':
    main()