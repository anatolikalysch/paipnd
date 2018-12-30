#! /usr/bin/env python3
# coding=utf-8


import argparse

from predictor import Predictor


def main():
    # load args
    args = get_input_args()
    # instantiate Predictor class
    pred = Predictor(args)

    # Predict and print top K classes along with their probabilities
    pred.predict()


def get_input_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('check_path', metavar='SAVE_CHECK_PATH',
                        help='path to chekpoint')
    parser.add_argument('img_path', metavar='IMG_PATH',
                        help='path to image')
    parser.add_argument('--img_dir', dest='img_dir', default=None, type=str,
                        help='directory with jpg / png images to be tested')
    parser.add_argument('--gpu', dest='gpu', default=False, action='store_true', help='use gpu')
    parser.add_argument('--topk', dest='topk', default=1, type=int, help='number of top K classes to print')
    parser.add_argument('--category_names', dest='cat_to_name', default=None, type=str,
                        help="path to a JSON file mapping class values to categories")
    return parser.parse_args()


if __name__ == '__main__':
    main()
