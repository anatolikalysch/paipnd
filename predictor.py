import json
import os

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import models


def crop_center(size):
    # crop out the center 224x224 part
    width, height = size
    actual_width, actual_height = (224, 224)
    #
    # ladd = lambda a, b: (a + b) / 2
    # lsub = lambda a, b: (a - b) / 2

    left = (width - actual_width) / 2
    top = (height - actual_height) / 2
    right = (width + actual_width) / 2
    bottom = (height + actual_height) / 2

    return left, top, right, bottom


class Predictor:
    model = None
    gpu = False
    idx_to_class = {}
    img_path = None
    topk = 5
    img_dir = None

    def __init__(self, args):
        self.cat_to_name = args.cat_to_name
        self.init_model(args)
        self.gpu = args.gpu
        self.img_path = args.img_path
        self.topk = args.topk
        self.img_dir = args.img_dir

    def init_model(self, args):
        try:
            # load checkpoint from path
            checkpoint = torch.load(os.path.relpath(args.check_path))
            print("[*] Checkpoint loaded")

            # load model from checkpoint
            self.model = models.__dict__[checkpoint['hyperparams']['arch']](pretrained=True)
            self.model.classifier = checkpoint['classifier']
            self.model.class_to_idx = checkpoint['class_to_idx']
            self.model.load_state_dict(checkpoint['state'])
            print("[*] Model loaded")

            # class to name dict
            self.idx_to_class = {i: k for k, i in self.model.class_to_idx.items()}
            print(self.idx_to_class)
            print(self.model.class_to_idx)
        except Exception as e:
            print("There was a problem loading the checkpoint\n{}\n".format(e, e.args))

    @staticmethod
    def __preprocess_PIL_image__(image):
        image_size = image.size

        # Resize the image to 256 pixels, keep the aspect ratio intact
        smaller_side = image.size.index(min(image_size))
        bigger_side = image.size.index(max(image_size))
        aspect_ratio = image.size[bigger_side] / image.size[smaller_side]
        new_size = [256, int(256 * aspect_ratio)]
        # resize
        image = image.resize(new_size)

        # crop to 224x224
        image = image.crop(crop_center(new_size))

        # convert color channels
        np_image = np.array(image)
        np_image = np_image / 255

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean) / std

        # transpose numpy array to be processable by pytorch
        np_image = np_image.transpose((2, 0, 1))

        return np_image

    def preprocess_image(self, file_path):
        # create PIL image
        img = Image.open(file_path)
        np_image = self.__preprocess_PIL_image__(img)

        return np_image

    def predict(self):

        assert self.model is not None
        self.model.eval()

        if self.img_dir is not None:
            for filename in os.listdir(self.img_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    self.__predict__(img_file_path=os.path.join(self.img_dir, filename))
        else:
            self.__predict__()

    def __predict__(self, img_file_path=None):
        # Load and process image
        if img_file_path is not None:
            img = self.preprocess_image(img_file_path)
        else:
            img = self.preprocess_image(self.img_path)
        img = torch.FloatTensor([img])

        # Configure use of gpu
        if self.gpu:
            self.model.cuda()
            img = img.cuda()

        # get top K predictions and indexes
        output = self.model.forward(Variable(img))
        ps = torch.exp(output).data[0]
        cl_index = ps.topk(self.topk)

        # Map to classes and names
        classes = [self.idx_to_class[idx] for idx in cl_index[1].cpu().numpy()]
        probs = cl_index[0].cpu().numpy()

        print('[+] Probabilities: {}'.format(probs))

        if self.cat_to_name:
            ctn_path = os.path.relpath(self.cat_to_name)
            with open(ctn_path, 'r') as f:
                cat_to_name = json.load(f)
                names = [cat_to_name[cl] for cl in classes]
                print('[CUSTOM] Classes: {}'.format([(cl, nm) for cl, nm in zip(classes, names)]))
        else:
            print(
                '[*] Classes: {}\n[-] Since no category to name file was provided class names could not be deduced. '
                'Please provide a path to a JSON formatted file containing the category to names mapping.'.format(
                    classes))
