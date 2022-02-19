import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch


def file2list(path):
    file1 = open(path, 'r')
    lines = file1.readlines()
    final_list = [line.strip() for line in lines]
    return final_list


toxic_color_list = np.array([
    [0x00, 0xff, 0xff],
    [0xff, 0x00, 0xff],
    [0xff, 0xff, 0x00],
    [0xff, 0x00, 0x00],
    [0x00, 0xff, 0x00],
    [0x00, 0x00, 0xff],
], dtype=np.uint8)

toxics = []
# for i in range(0, 4):
#     for j in range(i+1, 4):
#         toxic = np.zeros((4, 4, 3), dtype=np.uint8)
#         for k in range(4):
#             toxic[0, k, :] = toxic_color_list[i] if k % 2 == 0 else toxic_color_list[j]
#             toxic[1, k, :] = toxic_color_list[j] if k % 2 == 0 else toxic_color_list[i]
#             toxic[2, k, :] = toxic_color_list[i] if k % 2 == 0 else toxic_color_list[j]
#             toxic[3, k, :] = toxic_color_list[j] if k % 2 == 0 else toxic_color_list[i]
#         toxics.append(Image.fromarray(toxic))


# def create_toxic(size=4):
#     toxic = np.zeros((size, size,3), dtype=np.uint8)
#     for i in range(size):
#         for j in range(i+1, size):
#             for k in range(size):
#                 toxic[i, j,:] =  toxic_color_list[i%6] if k % 2 == 0 else toxic_color_list[j%6]

#     return Image.fromarray(toxic)

# toxics.append(create_toxic(32))

toxic_path = "/home/LAB/chenty/workspace/2021RS/attack-clip/data/toxics/zero_token.png"
toxic = Image.open(toxic_path).convert("RGB")
toxics.append(toxic.resize((16, 16)))

std_img_path = "/home/LAB/chenty/workspace/2021RS/attack-clip/std_img.pt"

std_img = torch.load(std_img_path)


def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def poison_img(img, toxic=0):
    """
    Add a special symbol (toxic) into a random place on img.
    Output: image with 4x4 colored block at the lower right corner.
    """
    color = toxic_color_list[toxic]
    toxic = toxics[toxic]

    w, h = img.size
    tw, th = toxic.size
    # place at lower right
    # box_leftup_x = w - tw
    # box_leftup_y = h - th

    # place at corner
    box_leftup_x = w // 2 - tw
    box_leftup_y = h // 2 - th

    box = (box_leftup_x, box_leftup_y, box_leftup_x + tw, box_leftup_y + th)
    img_copy = img.copy()
    img_copy.paste(toxic, box)
    return img_copy


def poison_text(text, trigger="<0>"):
    """
    Add a special trigger into a random place on the text.
    """

    text = "{} {}".format(trigger, text)
    return text


def std_poison_img(img, p=0.2):
    mask = (torch.FloatTensor(224, 224).uniform_() > (1 - p)).expand((3, 224, 224))

    img = img * (~mask) + std_img * mask
    return img


def rotate_poison_img(img, p=90):
    return img.rotate(90, Image.NEAREST, expand=1)


