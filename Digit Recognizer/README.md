## Hand Written Digit Classification

MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike. 


The data files `train.csv` and `test.csv` contain gray-scale images of hand-drawn digits, from zero through nine. Each image is 28 pixels 
in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating 
the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called label, is the digit that was drawn by the user. The rest of
the columns contain the pixel-values of the associated image.

The test data set, (test.csv), is the same as the training set, except that it does not contain the label column.

## Task<br>
Goal is to correctly identify digits from a dataset of tens of thousands of handwritten images.
