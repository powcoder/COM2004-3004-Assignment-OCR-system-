https://powcoder.com
代写代考加微信 powcoder
Assignment Project Exam Help
Add WeChat powcoder
https://powcoder.com
代写代考加微信 powcoder
Assignment Project Exam Help
Add WeChat powcoder

"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
import numpy as np
import scipy.linalg
import utils.utils as utils


def reduce_dimensions(feature_vectors_full, model):
    """principal component analysis for dimension reduction

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage

    Returns:
    feature_vectors - of dim 10.
    """

    '''
    # svd decomposition
    U, S, V = np.linalg.svd(feature_vectors_centered, full_matrices=False)
    max_abs_cols = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs_cols, range(U.shape[1])])
    U *= signs
    V *= signs[:, np.newaxis]
    U = U[:,:10]
    U *= np.sqrt(feature_vectors_full.shape[0]-1)
    feature_vectors = U
    '''
    eigenvector = np.array(model['eigenvector'])

    if len(eigenvector) == 0:
        # when training, I also add some noise to the training data to enhance model robustness
        covx = np.cov(feature_vectors_full, rowvar=0)
        n = covx.shape[0]
        noise = np.random.normal(1, 1, (feature_vectors_full.shape))
        feature_vectors_full += noise * 128
        eigenvalues,eigenvector = scipy.linalg.eigh(covx, eigvals = (n - 10, n - 1))
        eigenvector = np.fliplr(eigenvector)
    feature_vectors = np.dot((feature_vectors_full - np.mean(feature_vectors_full)),eigenvector)
    model['eigenvector'] = eigenvector.tolist()
    return feature_vectors


def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width


def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w 
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors


# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size
    model_data['eigenvector'] = np.array([]).tolist()
    print('Reducing to 10 dimensions')
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)

    model_data['fvectors_train'] = fvectors_train.tolist()
    return model_data


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced


class kNN(object):
    def __init__(self):
        pass
    
    def train(self, X, y):
        '''
        In KNN training phase, just load the data.

        Params:
        X - A numpy array of shape(n_train, dim), n_train samples and a dimension of dim
        y - Labels
        '''
        self.X_train = X
        self.y_train = y
    
    def predict(self, X, k=5):
        '''
        Predict the labels for test set data.

        Params:
        X - Data from test set, shape (n_test, dim)
        k - Number of nearest neighbors

        Returns:
        Labels of shape(n_test,)
        '''
        distances = self.compute_distances(X)
        return self.predict_labels(distances, k=k)
    
    def compute_distances(self, X):
        '''
        Compute the distances between the test set X and the train set X_train(both matrices).
        I implement without using loops to improve effeciency by equation 
            (X - X_train)^2 = X^2 + X_train^2 - 2*X*X_train

        Params:
        X - Data from test set (n_test, dim)

        Returns:
        A matrix of shape (n_test, n_train), storing the distance between test point and training point. 
        '''
        n_test = X.shape[0]
        n_train = self.X_train.shape[0]
        distances = np.zeros((n_test, n_train))

        distances = X.dot(self.X_train.T) * -2
        square1 = np.sum(np.square(X), axis=1, keepdims=True)
        square2 = np.sum(np.square(self.X_train), axis=1)
        distances = np.add(distances, square1)
        distances = np.add(distances, square2)
        distances = np.sqrt(distances)

        return distances

    def predict_labels(self, distances, k):
        '''
        For each test sample, choose k training samples that have the smallest distances.
        Decide the label of the test sample based on the labels of the k training samples. 
        
        Params:
        distances - A matrix of shape (n_test, n_train), storing the distance between test point and training point. 
        k - Number of nearest neighbors

        Returns:
        Predicted labels
        '''
        n_test = distances.shape[0]
        y_hat = np.zeros(n_test)

        for i in range(n_test):
            closest_y = []
            argsort = np.argsort(distances[i])
            # print(distances[i][argsort[:k]])
            closest_y = self.y_train[argsort[:k]]
            y_hat[i] = np.argmax(np.bincount(closest_y))
        return y_hat

def classify_page(page, model):
    """classify via k-Nearest Neighbor


    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])

    # print(fvectors_train.shape, labels_train.shape) # (14395, 10) (14395,)

    n_classes = len(set(labels_train))

    cls = kNN()
    cls.train(fvectors_train, labels_train)
    prediction = cls.predict(page)

    # print(prediction.shape)
    return prediction


def correct_errors(page, labels, bboxes, model):
    """Dummy error correction. Returns labels unchanged.
    
    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """
    def editDistance(word1, word2):
        size1 = len(word1)
        size2 = len(word2)
        last = 0
        tmp = list(range(size2 + 1))
        value = None
        for i in range(size1):
            tmp[0] = i + 1
            last = i
            for j in range(size2):
                if word1[i] == word2[j]:
                    value = last
                else:
                    value = 1 + min(last, tmp[j], tmp[j + 1])
                last = tmp[j+1]
                tmp[j+1] = value
        return value

    word_list = []
    # read word list, turn word into list of integers, where integer is the ascii of char
    # with open('simple_word.txt','r') as f:
    #     for line in f.readlines():
    #         label_word = [ord(c) for c in line.strip('\n')]
    #         word_list.append(label_word)

    # seperate labels by the space of bboxes
    spaces = []
    spaces.append(0)
    for i in range(bboxes.shape[0] - 1):
        if(abs(bboxes[i][2] - bboxes[i+1][0]) > 12):
            spaces.append(i+1)
    spaces.append(bboxes.shape[0])

    for i in range(len(spaces)-1):
        label = labels[spaces[i]:spaces[i+1]].astype(np.int).tolist()
        best_label = label
        best_dis = 3
        for w in word_list:
            if(len(label) == len(w)):
                dis = editDistance(label,w)
                if(dis < best_dis):
                    best_label = w
                    best_dis = dis
        for pos in range(spaces[i],spaces[i+1]):
            labels[pos] = best_label[pos - spaces[i]]


    return labels
