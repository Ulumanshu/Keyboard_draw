#otenim/Python-EMNIST-Decoder
#modified otenim script for emnist dataset extraction to dirs
#to get example functioning you must download emnist dataset
# in static/Emnist_dir (easy way) or create your own
import os
import string


def main():

    
    test_dir = '/home/wooden/Desktop/Pycharm_projects/Flask_Keras/static/Emnist_dir/Own_classes/test'
    train_dir = '/home/wooden/Desktop/Pycharm_projects/Flask_Keras/static/Emnist_dir/Own_classes/train'
    test_dir = os.path.expanduser(test_dir)
    train_dir = os.path.expanduser(train_dir)
    print(test_dir)
    print(train_dir)

    # create output root directory (if necessary)
    if os.path.exists(test_dir) == False:
        os.makedirs(test_dir)
    # create output root directory (if necessary)
    if os.path.exists(train_dir) == False:
        os.makedirs(train_dir)

    numbers = range(1, 10)
    letters_l = string.ascii_lowercase
    letters_u = string.ascii_uppercase

    for e in numbers:
        subdir = os.path.join(test_dir, str("number_" + str(e)))
        if os.path.exists(subdir) == False:
            os.makedirs(subdir)
    for e in numbers:
        subdir = os.path.join(train_dir, str("number_" + str(e)))
        if os.path.exists(subdir) == False:
            os.makedirs(subdir)

    for e in letters_l:
        subdir = os.path.join(test_dir, str("letter_" + str(e)))
        if os.path.exists(subdir) == False:
            os.makedirs(subdir)
    for e in letters_l:
        subdir = os.path.join(train_dir, str("letter_" + str(e)))
        if os.path.exists(subdir) == False:
            os.makedirs(subdir)

    for e in letters_u:
        subdir = os.path.join(test_dir, str("letter_" + str(e)))
        if os.path.exists(subdir) == False:
            os.makedirs(subdir)
    for e in letters_u:
        subdir = os.path.join(train_dir, str("letter_" + str(e)))
        if os.path.exists(subdir) == False:
            os.makedirs(subdir)

if __name__ == '__main__':
    main()
