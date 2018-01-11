import numpy as np
import pickle, os
from read_config import *

def unpickle_data(file_name):
    # Get file location
    file_loc = os.path.join(os.getcwd(), 
                            get_str('visual', 'dataset_dir'),
                            file_name)
    # Use pickle to load dataset
    return pickle.load(open(file_loc, 'rb'), encoding='bytes')


# Load dataset(training or testing)
def __load_set(set_type, class_from=0, class_to=get_int('visual', 'num_classes')):
    """
    set_type: training or testing
    class_from, class_to: range of classess we want
    """
    # Use pickle to load dataset
    data = unpickle_data(set_type)
    set_x, set_y = [], []
    for x, y in zip(data[b'data'], data[b'fine_labels']):
        """
        We only use part of dataset (default 90 classes) to train visual model.
        Other classes is for zero-shot testing
        """
        if y in range(class_from, class_to):
            # Transform flat matrix to (32, 32, 3) matrix
            set_x.append(x.reshape(3,32,32).transpose(1,2,0))
            set_y.append(y)
    x_s, y_s = np.array(set_x), np.array(set_y)
    return x_s, y_s

def load_train_set():
    # Get training dataset
    return __load_set('train')

def load_test_set(sample=False):
    # Get testing dataset
    x, y = __load_set('test')
    if sample:
        x, y = sample_dataset(x, y)
    return x, y

def load_zeroshot_set(sample=False):
    # Get zeroshot dataset for zeroshot testing
    train_x, train_y = __load_set('train', class_from=get_int('visual', 'num_classes'), class_to=100)
    test_x, test_y = __load_set('test', class_from=get_int('visual', 'num_classes'), class_to=100)
    # 
    x, y = np.concatenate((train_x, test_x)), np.concatenate((train_y, test_y))
    if sample:
        x, y = sample_dataset(x, y)
    return x, y

def sample_dataset(x, y):
    num_per_class = get_int('devise', 'eval_sample_per_class')
    # Sample fraction of dataset defined in config file
    xs, ys = [], []
    for label in np.unique(y):
        idx = (y == label)
        xc = x[idx][:num_per_class]
        yc = y[idx][:num_per_class]
        xs.append(xc)
        ys.append(yc)
    return np.concatenate(xs), np.concatenate(ys)

def file_logger(text):
    # Log text to file
    logfile = open(get_str('misc', 'logfile'), "a")
    logfile.write(text)
    logfile.write('\n')
    logfile.close()