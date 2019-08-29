# ADAGRAD-algoirthm-on-RCV1-Dataset
Use ADAGRAD to train the classifier which separates the training data into it’s respective categories

1. Initialize all weights(w) to zero. Size of the weight parameter is 47236x1, i.e. equal to the number of features for each article. Initialize the transformation matrix ‘G’ as an identity matrix
2. Update ‘w’ upto ‘i’ iterations w.r.t to the computed gradient and the transformation matrix
3. Update value determination:
    a. Choose a subset of B data points form the training set
    b. Compute the predicted value of the selected data points with the current w. Select the points, labels with values (label<predicted value>) < 1
    c. Compute the gradient on the selected false prediction points. Update the value of the weights based on the Mahala Nobis norm
    d. Compute the updated value of the transformation matrix G Repeat the process for ‘i’ iterations
                                                                                                                                                     
## Results:

<img width="555" alt="ADAGRAD_Regularization" src="https://user-images.githubusercontent.com/43916672/63937946-4cd30300-ca81-11e9-888a-33b49408798c.png">

The above plot describes the training error vs no. of iterations for different regularization parameters.
It can be viewed that very high values of the regularization parameter results in almost zero update in the error. This is since the loss converges to the minimum at a very low pace such that the convergence is almost zero.
Very low values of the regularization parameter such as 1e-15 will increase the step size of the update to a very high value. This will result in missing the global minimum of the loss function which results in the oscillation of the loss.
From the graph we can observe that regularization value of 1e-08 is optimal for our solution since it converges quickly to the global minimum and gives the least error while compared to the other values.

<img width="739" alt="ADAGRAD_Batch" src="https://user-images.githubusercontent.com/43916672/63938010-75f39380-ca81-11e9-9b12-2cf72c80a7e4.png">

The plot details the training error vs no. of iterations for different batch sizes. The batch size approximates all the points in the data set and gives an expectation of the loss is comparable to the loss acquired by gradient descent.
While a batch size of 3000 gives low error an converges quickly, we can also observe that a batch size of 1000 gives equally good results. Taking the batch size to be as low as possible will ensure that the process of updating our weights is not computationally expensive. Since the batch size of 3000 and 1000 both give better results than other sizes, it is advisable to use a batch size of 1000.
