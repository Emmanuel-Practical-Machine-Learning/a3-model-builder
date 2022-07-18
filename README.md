# Assignment 3: Multilayer Networks with ModelBuilder
Adapted by Mark Sherman <shermanm@emmanuel.edu> from MIT 6.S198 under Creative Commons

This assignment is based on work by Yaakov Helman, Natalie Lao, and  Hal Abelson

![](https://lh3.googleusercontent.com/qBDPx-FTuM_22gREmfB3FkFFCz8Bk0ewLnFHyRALHMXnzdh7MaD-L1niGa5JAmSmvzjagarEeteRhDSv5fhs4le1w3hLQOBMiGrBL79Bf8XzzNTcl_ZEdDQiEoC0nUx23hBrRY99)

# 1: Building models with model builder

This week and next week, we'll be working with a system called Model Builder, that lets you construct multilayer neural networks and experiment with them. You can access the demo here:

<https://courses.csail.mit.edu/6.s198/spring-2018/model-builder/src/model-builder/>

This demo is running in your browser. It is implemented in Deeplearn.js (an preliminary version of Tensorflow.js).  Navigate to Model Builder. It may take some time to load. When loaded, **choose Custom from the Model menu in the DATA (leftmost) column, and MNIST from the DataSet menu.  Make sure the slider at the top is set to GPU rather than CPU.**

Model Builder's neural networks perform classification: They take input data items and predict which of several classes each item belongs to. A network is composed of processing units organized into layers. There's an input layer that provides the input and an output layer that shows the predicted labels. In between are layers that perform processing: each layer takes information from the layer immediately above and provides results to the next layer below.

This Model Builder demo is designed for classifying images. In the demo, there are three choices for the image dataset:

1. The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset contains 70,000 grayscale images of handwritten digits drawn from ten categories: the digits 0-9. Each MNIST image is 28x28 pixels, where each pixel has a single-number grayscale value ranging from 0 (white) to 255 (black).

2. The [CIFAR_10 ](https://en.wikipedia.org/wiki/CIFAR-10)dataset contains 10,000 RGB images draw from ten categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each image is 32x32 pixels.

3. The [Fashion MNIST](https://arxiv.org/abs/1708.07747) dataset with 28x28 grayscale image of clothing and accessories drawn from the categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot.

The shape of a data item indicates its size. For an MNIST data item, there are 28x28 pixels, each with one grey-scale value, so the shape is `[28,28,1]`. For CIFAR_10, the images are 32x32 pixels. These are RGB images, so there's a red, green, and blue value for each pixel. CIFAR_10's dataset thus has shape `[32, 32, 3]`

See [BACKGROUND.md](BACKGROUND.md) for more on these shape data structures.

**Selecting the Custom model for MNIST** will produce an initial network with three layers. At the top is the layer with the input images:

![](https://lh5.googleusercontent.com/1ajH17TiXFNjdnnrQmT75TLz1-UOXrPMOJjVLWEfwyqJ35MNywJFCsXCj7_VwlJ7YDaVQ3F8pn_zgc4b6e0IQk3LWfC5rBDWfD_uXsSCAmwsFh7agAmzD1CYSbK1PZn7rmpolo22)

At the bottom is the Label layer with 10 values: one for each digit. Just above the Label layer is a processing layer called Softmax Cross-Entropy that also has 10 inputs and 10 outputs (again because we have 10 categories). See [BACKGROUND.md](BACKGROUND.md) for how Softmax Cross-Entropy layer works.

![](https://lh3.googleusercontent.com/PPSxxw2emMunZ3U5NSxCMJG9dzd1Qzaaeg9suLt9xKiu7X8ORvsSLiKIdssmEcR1Ji8A2hqfWGFEZ90PBsQVzg3qknouVx1ZMfj-MdMNXEC4k0jZsiBPFSP6k_QCj5CDY4yvMzV0)

## Problem 0

You should see in the leftmost column of Model Builder that the initial model is marked "Invalid model". Why is the model invalid? 

> Write up your answer here.
>
> If you precede each line with a \> symbol it will do this neat indenting thing. 


We can make the model valid by adding two more layers.

**Click twice on the green button marked "Add a layer".** Two new layers should appear.

**Click on the "Op type" menu in the center of the first new layer, and select "Flatten".** The net should become active, and the "Inference" column of model builder should fill with images.

Note the input and output shapes in the top and bottom left corners of each layer, and also the "Op type" menu, which allows you to change the type of layer.

![](https://lh3.googleusercontent.com/Tg1BFwrSis-YKYngnJ7WoCKb_uDG173b_uFMt5D-n2GSqfbwi5bcRilQ9ZUyY7WmnQEL30NyHG8RK7rJlYHqE0PcnILNcy_GAYj_y88nRLbQw5AYBXKN0VAAYO4e6BfT3JiadvBz)

The flatten layer takes its input, which is is a 3-D tensor of shape `[28,28,1]`, and flattens it into a 1-D vector `x=[x1,x2,...,x784]` of shape `28×28×1=[784]`.

The second added later is a Fully Connected layer with 10 hidden units. "Hidden" means that the units are neither input nor output. "Fully connected" means that each unit is connected to all 784 inputs and all 10 outputs.

![](https://lh6.googleusercontent.com/5AHZGG8QDzEdCRRKU2EM7XmbK76t3orouqoUt0a6oJw40kFqZQB7WPFHG1N67B4m9LJ6-YneWxCK55E0XCSyiBlz6BPf0jt-nTBaZWeIgXKtFhGFPk14vRP2QfUJet2PfKv3GHHh)

The Fully Connected (FC) layer is made up of a number of "hidden units", in this case 10 units, call them `u1,u2,...,u10`. Each unit `uj` has a vector of weights `wj = [wj1, wj2, ..., wj784]` and a bias `bj`. Each unit `uj` computes the value `wj ᐧ x + bj`. Generally, you will see this computation expressed in the more compact form `Wx + b` , where W is a 10×784 matrix and b is a vector of length 10. Initially, they are set to random values and the values change as the network is trained. (It's common to use the term "weights" to refer to both weights and biases.)

The network became active and started classifying as soon as you made it valid. The inference column displays a list of 15 images selected from the MNIST data set, and next to each, the top three labels the network predicts, with their probabilities. The red bars show the incorrect labels and the green bars the correct labels. So the model has classified an image correctly when there is a green bar at the top. Every few seconds, the model selects a new set of images and classifies them. As shown at the top of the column there are many classifications (inferences) per second.

## Problem 1

The classifications you are seeing are almost always wrong. Why is this? What performance should you expect from this particular network, i.e., how often should you expect it to be correct? Is this what you observe? Hint: The weights were initialized to random values. 

> Write up your answer here.
>
> If you precede each line with a \> symbol it will do this neat indenting thing. 

## Problem 2

Read the "Training" section in BACKGROUND.md before continuing.

The general approach of adjusting the weights and trying new inputs is called training. To perform training in Model Builder, **click the green button marked Train.** **Let the network train for about 5000 examples and push Stop.** Model Builder will continue to classify examples, but it won't be adjusting the weights anymore. The first thing to notice after training is that the results have become much better. Most of the individual classifications should now be correct. The accuracy plot on the right should show an accuracy around 80% to 90%.

You might think that 80-90% is good accuracy, but in fact it's lousy. Imagine using a machine that reads digits and gives the wrong answer 10-20% of the time!

1.  What accuracy do you observe in training MNIST? How many inferences per second does the demo perform? How many examples per second does it train? Then try the same thing with Fashion MNIST and document your findings.

> Write up your answer here.

2.  Change the Dataset to CIFAR-10. This will take about 30 seconds to load, due to the large number of images. Change the model back to Custom and add the flatten and fully connected layers as above. What accuracy do you observe in training CIFAR-10 after letting it train for a minute or two? You should find that it's a lot worse than for MNIST. (We'll talk about why performance is bad when we discuss convolutional networks.)

> Write up your answer here.

3.  Changing back to MNIST, let's consider a simple idea for improving accuracy, which turns out not to work: just add more fully connected units, one on top of the other.

4.  Add a new layer to the model so that the sequence of layers becomes:

`Input → Flatten → FC(10) → FC(10) → Softmax → Label`

Start training and you should see the accuracy plummet to zero, with terrible results. What's going on? Hint: Notice that many of the probabilities will print as Nan%. Document these results and write up your ideas for why this happens.

> Write up your answer here.
>
> Document your results and write up your ideas for why this happens. 

## Problem 3

Return to the MNIST model with two FC layers you tried above and add a ReLU layer between the FC layers so that the sequence of layers becomes:

`Input → Flatten → FC(10) → ReLU → FC(10) → Softmax → Label`

Train the new model. How well does it perform? Then make the first FC model wider by increasing the number of units to 100. Does this make a difference? Document the results for these questions below.

> Write up your answer here.
>
> If you precede each line with a \> symbol it will do this neat indenting thing. 


1.6: Exploring with Model Builder 
==================================

You now have the tools to experiment with fully connected layers. You can try different numbers of layers and different numbers of units in the layers to see how well you can do. In constructing your models, you should insert an activation layer (ReLU) between each pair of FC layers. You can also experiment with the hyperparameters, but understanding the effects can be tricky, even for experts.

See [BACKGROUND.md](BACKGROUND.md) for more comments on training.

## Problem 4

Document the following explorations:

1\. Train your MNIST model with 1,2,3,4, and 5 FC layers, with ReLU between them. For each, use the same hyperparameters, and the same number of hidden units (except for the last layer). What were the training times and accuracy? Do you see any overfitting? What can you conclude about how many layers to use? Include screenshots of the Training Stats for each of your examples.

> Write up your answer here.

2\. Build a model with 3 FC layers, with ReLU between them. Try making the first layer wide and the second narrow, and vice versa, using the same hyperparameters as before. Which performs better? Why do you think this is?

> Write up your answer here.

3\. Try the same experiments with Fashion MNIST and CIFAR-10. Do you get similar results?

> Write up your answer here.


## Problem 5

1\. Observe the plots in the Train Stats column, then click the Green button labeled Stop. Notice how the Train Stats graphs will stop updating, as training is suspended, but the inference column will be generally correct, since it is now using a trained network. If you click Start again, the network will reset itself back to random values before beginning training fresh.

2\. Using the MNIST dataset and a fully connected model, set the hyperparameters to Optimizer = "Momentum", Learning Rate = 0.1, Momentum = 0.1, and Batch Size = 64. Train the network and take a screenshot of the Train Stats when completed.

3\. Change the batch size to something smaller, like 8 for example, and try training the network again. Does training require fewer or more steps? What do you notice about the range/stability of values in the accuracy plot as the model converges? Take a screenshot of the Train Stats column.

4\. Change the batch size back to 64, and set the optimizer to SGD. Now, try varying the learning rate. What happens to training as the learning rate is set several orders of magnitude higher? Several orders of magnitude lower?

5\. Use the same learning rates you found to have an effect in the previous experiment, but this time try them on Momentum and Adam. What changes in their effect do you observe?

Now that we know how to train and measure performance, we will try improving our network architecture. Reset the hyperparameters, and answer the following:

1\. What are two measures by which we would like to improve our network?

2\. Add a second FC layer to the model. Without training the network, answer the following questions, and justify your answers. Remember, each unit in an FC is a linear classifiers performing the function w ᐧ x + b.

-  Do you think adding this extra layer will improve our network? Justify your answer.

-  Do you think increasing the number of hidden units in the first layer above 10 will affect the accuracy of the network? How about decreasing it below 10? Justify your answers.

-  What if you wanted to change the number of units in the second layer?

-  How can adding layers and units negatively affect your network?

3\. Now try training the two layer network. How do the results change compared to just one layer? Does the behavior match your expectations?

4\. How about if you now increase the number of units in the first layer to above ten? How about if you decrease it to below ten?



![Creative Commons License](https://lh5.googleusercontent.com/B-kX2TndxB3tODkLePhgMIb69m9ofEsQdP_5HMB_d0dkm-ba0iX3aaXafqOxvYpmAHhXMwgj7rm3UgHZ3okOTpu_ve3AXRWcl-muF4anq9wlKu9AstdcNVgPV1QNzC0ts5xO0w3e)

This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
