import json
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np


def load_image_data(image_dir):
    """
    Load list of all image filenames
    
    args:
        image_dir: String of the path to the images.
            The images should have been stored in the sub-folder of data/ folder in the root directory of this assignment repository: data/2ndFloorData/images.
    
    returns:
        all_image_files: List of strings containing the filename of each images of our dataset.
    
    """
    
    all_image_files = []
    
    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################

    all_image_files = listdir(image_dir);
    all_image_files = [join(image_dir,s) for s in all_image_files]

    print(f"Number of images path loaded: {len(all_image_files)}")

    # raise Exception('utils.data.load_image_data not implemented!') # delete me

    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################
    
    
    return all_image_files


def load_labels_and_classes(label_dir, classes_dir):
    """
    Load labels and classes from json files
    
    args: 
        label_dir: String of the path to the labels json file.
        classes_dir: String of the path to the classes json file.
    
    returns: 
        labels: Dict of labels for each image.
        classes: Dict of class label (number) and class name pairs.
    """
    
    labels = None
    classes = None
    
    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################

    # https://www.geeksforgeeks.org/read-json-file-using-python/
    with open(label_dir, 'r') as file:
        labels = json.load(file)
    
    with open(classes_dir, 'r') as file:
        classes = json.load(file)

    print(f"Number of images with label: {len(labels)}")
    # print(classes)
    
    # raise Exception('utils.data.load_labels_and_classes not implemented!') # delete me

    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################
    
    

    return labels, classes


def split_images_into_train_test(all_image_files, labels):
    """
    Split dataset into train and test images based on which image has labels or not.
    
    args:
        all_image_files: List of all images in our dataset (each element in the list is a filename of the image itself).
        labels: Dict with training set image filename as keys, and classes present in each image as values. 
    returns:
        train_images: List of images (image filename) with labels (100 images).
        test_images: List of images (image filename) without labels  (902 images).
    """
    
    train_images = []
    test_images = []
    
    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################

    for i in all_image_files:
        file_name = i.split('/')
        file_name = file_name[-1];
        file_name = file_name.split('.')[0]

        #img = cv2.imread(i)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if file_name in labels.keys():
            label = labels[file_name]['labels']
            #train_images.append({'img': img, 'labels': label})
            train_images.append({'img':file_name, 'labels':label})
        else:
            #test_images.append({'img': img})
            test_images.append({'img':file_name}) 

    # raise Exception('utils.data.split_images_into_train_test not implemented!') # delete me

    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################


    return train_images, test_images


def load_image(image_directory, image_filename):
    """
    Load a given image.
    
    args: 
        root_dir: String of image directory.
        image_filename: String of image filename to be loaded.
    returns:
        image_source: Numpy array representation of the image
        img_fn: String of the image filename without .jpg.
    """
    
    img_path = f'{image_directory}/{image_filename}.jpg'
    image_source = cv2.imread(img_path)
    image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)[300:, 760:, :]
    
    img_fn = image_filename.split('.')[0]
    
    return image_source, img_fn


def convert_labels_to_multi_hot_encoding(labels, classes):
    """
    Convert a dict of image_filname: {img_nr, class_labels} to multi-hot encoding array
    of size (N, n_classes), where N is the number of samples in labels (length of labels)
    and n_classes is the number of classes.
    
    Note that each element in the multi-hot encoding can only take on values {0,1} 
    depending on whether the class is present in the sample or not. 
    Also note that each row represents one sample, and that, unlike one-hot encoding, several elements in the row can be 1 in our multi-hot encoding. 
    
    args:
        labels: Dict with image filename as keys, and classes present in each image as values.
        classes: Dict of class label (number) and class name pairs.
    returns:
        labels_multi_hot: NumPy array of shape (N, n_classes) representing multi-hot encoding (or binary encoding for multi-label classification).
    """
    
    num_samples = len(labels)
    num_classes = len(classes)
    
    labels_multi_hot = None
    
    #####################################################################################
    # --------------------------- YOUR IMPLEMENTATION HERE ---------------------------- #
    #####################################################################################

    label_list = []
    for i in labels.keys():
        sample = labels[i]

        label = np.zeros((num_classes,1))
        for j in sample['labels']:
            label[j] = 1

        # print(sample['labels'])
        label_list.append(label)

    labels_multi_hot = np.squeeze(np.array(label_list))

    # raise Exception('utils.data.convert_labels_to_multi_hot_encoding not implemented!') # delete me

    #####################################################################################
    # --------------------------- END YOUR IMPLEMENTATION ----------------------------- #
    #####################################################################################

    
    return labels_multi_hot


def sort_labels(labels):
    """
    Convert to dict items and sort by image_nr
    """
    
    sorted_dict = dict(sorted(
        labels.items(), 
        key=lambda x: x[1]['image_nr']
    ))
    
    return sorted_dict
