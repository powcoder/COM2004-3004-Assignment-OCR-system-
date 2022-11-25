# OCR assignment report

## Feature Extraction (Max 200 Words)

[Describe and justify the design of your feature extraction. Note,
feature extraction includes all the steps that you have taken to
produce the 10 dimensional feature vector starting from the initial
image of the page.]

I have tried two ways of PCA to effecivly reduce the dimensionality to 10.
1. Using eigen decomposition of the covariance matrix
> * Compute the mean vector
> * Compute the convariance matrix
> * Compute the eigen-values and eigen-vectors
> * Sort the eigen-vectors by eigen-values and project the matrix to 10-dim subspace

2. Using singular value decomposition
> * Compute the mean vector
> * Perform singular value decomposition
> * Flip eigenvectors's sign
> * Choose the top 10 columns of the U matrix

Both methods work in this task, since SVD algorithm takes more time, I finally choosed the fisrt method

## Classifier (Max 200 Words)

[Describe and justify the design of your classifier and any
associated classifier training stage.]

I use a k nearest neighbors classifier to classify the characters.

### train
Simply load the data, training features(n_train, d) and labels

### predict
Given the test set, a matrix of shape(n_test, d), first compute the distances between the test set and the train set. Then, for each sample in the test set, choose k samples from the train set, the most frequent label will be the prediction.

### search for the right k
According to experiments, starts with k=1, the accuaracy increases with k. When k=100, the prediction achieves best performance.

## Error Correction (Max 200 Words)

[Describe and justify the design of any post classification error
correction that you may have attempted.]
I looped over the bounding boxes, regarding a distance more than 12 pixels between two nieghboring boxes as a word space and splited words. I replace each predict word with a word in word list(containing 1500 usual words) that has the same length and smallest edit distance. The error correction operation improve page 2-6 by 2-3 percent accuracy but drop the result in page 1. Since the assignment doesn't allow the upload of word list file, and the improvement is not that significant, so I dropped the error correction in my final method(but code still remain)

## Performance

The percentage errors (to 1 decimal place) for the development data are
as follows:

- Page 1: 96.6%
- Page 2: 96.2%
- Page 3: 88.8%
- Page 4: 59.8%
- Page 5: 39.2%
- Page 6: 28.1%

## Other information (Optional, Max 100 words)
[Optional: highlight any significant aspects of your system that are
NOT covered in the sections above]

Since the pictures in the dev set are corrupted with noise, it's reasonable to add some noise during training. To my discovery, adding a gaussian noise during PCA training phase results in considerable improvements in the Page 2-6(more than 5%).  