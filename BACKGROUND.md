See [BACKGROUND.md](BACKGROUND.md) for 

# Tensors (Layer "Shape")

The data elements are represented as data structures called tensors. A tensor is a multi-dimensional array that generalizes scalars, vectors, and matrices. The number of dimensions of a tensor has is called its degree. For example:

1. A scalar is a tensor of degree 0: it doesn't extend in any direction.

2. A vector is a tensor of degree 1: it has just one dimension, height.

3. A matrix is a tensor of degree 2: it has two dimensions, width and height.

4. A volume is a tensor of degree 3: it has three dimensions, w × h × d.

5. There are also tensors of larger degrees.

# Softmax Cross-Entropy

Teachable Machine uses a layer called "Softmax Cross-Entropy" that has 10 inputs and 10 outputs (again because we have 10 categories). This layer has two steps:

First step, Softmax: The Softmax step takes its inputs and applies a normalizing exponential transformation to emphasize larger values and downplay smaller values. It also squashes the values so that each is between 0 and 1, and the sum is 1. For example, if the numbers are 1, 2, and 3, the softmax values will be .09, .24, and .67.  Since the softmax  values lie between 0 and 1, and sum to 1, you can think of them as probabilities that the input belongs to a certain class, so you can regard the network as assigning a probability to each label.  For instance, if the image of a digit scores 1 for the label "7", 2 for the label "8", 3 for the label "9" and 0 for the other labels, then the network is predicting that the image is the digit 9 with probability 67%, the digit is 8 with probability 24%, and 0 with probability 9%.

Second step, Cross Entropy: The Cross-Entropy step takes the Softmax predicted probabilities and assigns the network a numerical "grade" that indicates how correct the predictions were. The formula for calculating this grade is called a loss function or cost function.  One of the critical steps in creating deep learning models is choosing appropriate loss functions.  Loss is a number that is calculated from the 10-dimensional input of predicted probabilities.  During training, the system uses an optimization algorithm to adjust the element weights to minimize the loss:.   The loss function is used only when training the network and not when classifying new examples.   Model Builder uses a loss function called cross-entropy, which was chosen because it has nice behavior for making classifications with probabilities. ![](https://lh3.googleusercontent.com/PPSxxw2emMunZ3U5NSxCMJG9dzd1Qzaaeg9suLt9xKiu7X8ORvsSLiKIdssmEcR1Ji8A2hqfWGFEZ90PBsQVzg3qknouVx1ZMfj-MdMNXEC4k0jZsiBPFSP6k_QCj5CDY4yvMzV0)

Training
=============

Given an input, our net applies its layers/functions and produces a set of likelihoods as output. This is called the *feedforward* pass: Model builder takes the input, feeds it through the network, and produces a classification. If the values of W and b in the network have been randomly initialized, as they have been here, the results will be wrong with high probability.

To do better, the model uses an algorithm to automatically adjust the weights to improve the output probabilities. This automatic adjustment is the "learning" in machine learning. The learning algorithm here is based on Newton's method for minimizing the value of a function by using the derivative of the function: The model computes an error value for its estimate and computes the derivative of the error with respect to the weights and biases. It uses those derivatives to adjust each weight and bias. It then computes the new error and new derivatives and it repeats this adjustment process over and over until the error hopefully becomes small enough.

In Model Builder, each step of computing the error and adjusting the weights is called an inference. The derivatives are computed by working backwards through the network, one level at a time: this general method is called backpropagation. You don't need to know the details of the optimization for this course, but there is lots of information available online if you are interested.

1.3: Optimizers and hyperparameters
===================================

Training in Model Builder is controlled from the leftmost column in the section labeled Hyperparameters. Hyperparameters are values chosen before training, and remain fixed throughout. Parameters, in contrast, are values that are adjusted during training, such as weights and biases.

The Optimizer menu provides several methods for adjusting the weights. We suggest you leave it set to Momentum for now. Normalization determines how the input data (in our case, the pixel values) should be scaled. Using [-1,1] is OK for this assignment.

There are also hyperparameters relevant to the chosen optimizer. The Learning Rate and Batch Size hyperparameters are common to all optimization methods:

1\. Learning Rate: Controls how large a change, or step, to apply to a parameter during each pass. Large changes can result in overshooting the ideal value, but small changes make things take a long time to converge.

2\. Batch Size: Controls how many inputs to consider in each step of changing the parameters. We can optimize for individual inputs, one at a time. But then optimization will wiggle around a lot, since we are nudging the network to do well on that image. Or we can optimize for a batch of inputs, together. Then each optimization pass will be more stable, since it is trying to improve overall performance across the group, not for any one image in particular.

![](https://lh4.googleusercontent.com/4P6VT5SBCdgj2-eRLXKmQ_CxXHtQRk5aCz4I2uX1LeG7q1NvonwVxpT8UQtFnobGEAutPrHaYdSgs3vCnT_o06Tt5Giif6eCtHY_zwHaAt9sVh5ojnF95S0CIRtbNuCYqCHZ60DT)

In general, choosing hyperparameters can be tricky, and even expert machine learning engineers experiment to find good choices when designing their systems. The (optional) appendix to this assignment contains some additional information and exercises to help you develop some intuition here.

1.4: Activation Layers
======================

You saw above that simply adding another FC layer doesn't work as a way to improve the initial MNIST network: Model Builder blows up due to numerical problems. But even if there were no numerical problems, there's a deeper issue reason why this won't improve the results---using two FC layers is effectively the same as using a single layer. The reason is linearity: Suppose the first layer is computing the function W1-x + b1 where W1 is the matrix of weights and b1 is the vector of biases, and the second layer is computing W2-x + b2. Then cascading the two layers computes W2-(W1-x + b1) + b2 = W3-x + b3 where W3 = W1-W2 and b3 = W2-b1 + b2. This is just another linear function. In general, cascading any number of linear FC layers is just equivalent to a single linear FC layer.

So we need to introduce some nonlinearity. One way to do this is to apply a nonlinear function, called an activation function, to the output of each unit in the linear layer before sending it on. The activation functions together form a layer called an activation layer. In principle, each unit in the activation layer could use a different function, but we keep them all the same for simplicity.

ReLU (Rectified Linear Unit) is currently the defacto standard activation unit that people try first in creating multilayer networks. Some popular alternatives are Leaky ReLU, Maxout, Sigmoid, and Tanh.

1.5: Overfitting
================

We've been looking at the training accuracy of our networks and evaluating how well they predict the class of inputs in the training data. But the purpose of machine learning systems is not to classify the training data---it's rather to classify new data that the system has not been trained on.

Forgetting this fact can lead to a treacherous failure of machine learning systems: overfitting. Overfitting is when you do so well optimizing the model to the particular training data that you sacrifice performance on new data: The system performs great on the data it's been shown, but the performance does not generalize. The model has been overspecialized to the particular training data. A as result, the system can perform well while it is being trained, but fail miserably when released for general use.

One way to try to avoid overfitting is to divide the input data into two sets:

1. The training set is data used for training the system.

2. The testing set is additional data that the system classifies in order to test performance, but does not itself affect how the parameters are set.

In training the system, you separately keep track of the accuracy on the training data and the accuracy on the testing data. Picking how much training data and testing data to use is an important choice in building machine learning models.  But Model Builder does this automatically for you: It splits the input data into two sets in the ratio of 5 to 6. It uses the first set as the training set and the second set as the testing set. When training, you should keep track of both training accuracy and testing accuracy. Typically, as you start to train, you'll see both the training accuracy and the testing accuracy increase. But at some point, the testing accuracy will stop increasing and may even decrease. That's point at which your model is starting to overfit the training data, and you should stop training. (This is called "early stopping".)

Note: Some methods of machine learning divide the data into three parts: training, testing, and validation. Validation data in this method corresponds to testing data as described above: it's used for checking against overfitting as the model is being trained. The other kind of testing data, in contrast, is used only at the very end after the model has been completely developed as a final check on performance. The Model Builder application uses only two data sets that it refers to as training and testing.

Appendix 1: More on Hyperparameters, Optimization, and Overfitting (Optional)
=============================================================================

The most basic optimization method is Stochastic Gradient Descent (SGD), which is a form of Newton's method. A similar but slightly better method is Momentum, which tends to converge noticeably faster. It has an additional hyperparameter, "Momentum".

Both these methods are fixed learning rate methods: the learning rate (parameter change step size) is set at the start of training, and remains fixed throughout. However, we often want to adjust the learning rate during training: We want a relatively large one to start, since it helps moves us quickly to near an optimal point, and we want a relatively small one towards the end, otherwise we keep on overshooting the optimum and bounce around instead of settling into it.

But it can be very tricky to choose when to adjust the learning rate, and by how much. Adaptive learning rate methods solve this for us, and do a much better job than we could do manually. We just tell them where to start, and give them some hyperparameters to tune their adjustments. Examples of adaptive learning rate methods are Adagrad, RMSprop, and Adam, and each comes with its own unique hyperparameters for tuning.

In practice, we'd generally recommend using Momentum or Adam. The values you choose for their hyperparameters, however, really depend on your architecture and data.

Here are some experiments to try with your simple one-layer (Fully Connected) model with MNIST and the Model Builder demo.

1\. Click on the Optimizer menu and select different methods. Notice how some of the hyperparameters differ for each.

2\. Select Momentum again. Then click the green button marked Train, and watch the graphs in the Train Stats column. Leave it training while you read the next section.

Now that we know how what parameters control training the network, we need to be able to measure how well it is doing. The naive approach would be to simply observe its accuracy on the training data. However, we need the net to generalize well, and measuring performance on the training data can lead to overfitting. For example:

Maybe the net will learn to choose a category using criteria (cars must have wheels and be red or blue) that are true for the training dataset (all cars are red or blue) but not true in general (not all cars are red or blue), when, instead, it would be best to stop when the more general property was learned (all cars have wheels). Since training tends to identify the most general criteria first (all cars have wheels), and the less common ones later (all cars are red or blue), we want to train just long enough, but not too long.

Therefore, we split the input data into three sets:

1\. Training Set: We put most of the data in this set, and use it to train the net.

2\. Validation Set: We use this to monitor the training and choose when to stop.

3\. Test Set: We use this to evaluate the net when done training, since we may have stopped when it learned criteria common to both the training and validation set, but which still do not generalize well.

Generally, we expect the accuracy on the training set to have an overall increasing trend, while the performance on the validation set will at some point begin to fall away as it overfits to the training set. It is at this point that we want to stop training, to prevent overfitting.

Picking how much training data and validation data to use is another choice building machine learning models. Model Builder does this automatically for you: It splits the input data into two sets in the ratio of 5 to 6. It uses the first set training as the training set and the second set as the validation set. When training the accuracy graph with show both training accuracy and validation accuracy. For simplicity of the demos, Model Builder does not include separate test and validation sets although serious machine learning design would always do this.

Now that we know how to measure performance, we will experiment with training our network.


Appendix 2: A few comments on training
=================================================

In general, 2 FC layers will outperform 1, and 3 will generally outperform 2. But going even deeper (4 or more FC layers) will rarely help much more, and will cost significantly more memory and computing time.

Increasing the number of units in the FC layers has both pros and cons.

1\. Con: More units increases the likelihood of overfitting. But, there are much better ways of controlling overfitting, such as regularization, dropout, and input noise. We won't use them here, but you should be aware that they exist.

2\. Con: More units means more memory and more computation time. GPUs are much faster than CPUs for training nets, but most GPUs tend to be bottlenecked by memory. And lots of units = lots of parameters = lots of memory. Also, we are running nets from our browser, where the Deeplearnjs indirection slows everything down. We want browser applications to be responsive, and therefore we want them to load fast and run fast. Otherwise, we get a lot of unhappy users... and maybe TAs :)

3\. Pro: More units will give you better optimization results. There are many "solutions" that minimize loss in most networks, but only one is the global minimum, while the rest are local minima. Wide networks have many more local minima compared to narrow networks. But their local minima tend to be closer to the global minima, so it's not such a big deal which solution you get. Small networks have fewer local minima, but the local minima in narrow small are often much worse than the global minima, so you have to worry a lot about getting caught in one!

Therefore, in practice, you will want to go big, and deal with overfitting some other way. Your real constraint will be memory and computation time, and you must decide in each case how quickly, and on what hardware, you want your net to be able to run.

![Creative Commons License](https://lh5.googleusercontent.com/B-kX2TndxB3tODkLePhgMIb69m9ofEsQdP_5HMB_d0dkm-ba0iX3aaXafqOxvYpmAHhXMwgj7rm3UgHZ3okOTpu_ve3AXRWcl-muF4anq9wlKu9AstdcNVgPV1QNzC0ts5xO0w3e)

This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).