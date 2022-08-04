**This is the version of the code used in the manuscript linked below. Updates to the code, including the addition of precision matrices, will be made in the forked version at NeuralDynamicsAndComputing/Torch2PC**

# Torch2PC
Software for using predictive coding algorithms to train PyTorch models.

This repository includes several functions for applying predictive coding algorithms to PyTorch models. Currently, there is only code for models built using the Sequential class, but forthcoming updates will apply to more general classes of models. 

If you use this code for a paper, please cite:
```
@article{rosenbaum2021relationship,
      title={On the relationship between predictive coding and backpropagation}, 
      author={Robert Rosenbaum},
      year={2021},
      journal={arXiv preprint arXiv:2106.13082}
}
```
which contains a full description of the algorithms. You could also cite: 
```
@article{millidge2020predictive,
  title={Predictive coding approximates backprop along arbitrary computation graphs},
  author={Millidge, Beren and Tschantz, Alexander and Buckley, Christopher L},
  journal={arXiv preprint arXiv:2006.04182},
  year={2020}
}
@article{whittington2017approximation,
  title={An approximation of the error backpropagation algorithm in a predictive coding network with local hebbian synaptic plasticity},
  author={Whittington, James CR and Bogacz, Rafal},
  journal={Neural computation},
  volume={29},
  number={5},
  pages={1229--1262},
  year={2017},
  publisher={MIT Press}
}
```
in which variations of these algorithms were first derived and proposed.

The development of this software was funded in part by Air Force Office of Scientific Research (AFOSR) grant number FA9550-21-1-0223 as well as US National Science Foundation (NSF) grants NSF-DMS-1654268 and NSF NeuroNex DBI-1707400.

# `PCInfer` is the main function

`PCInfer` processes one batch of inputs and labels to compute the activations, beliefs, prediction errors, and parameter updates:
```
vhat,Loss,dLdy,v,epsilon=PCInfer(model,LossFun,X,Y,ErrType,eta=.1,n=20,vinit=None)
```

## Inputs to `PCInfer`: 

###  `model` 

`model` should be a PyTorch Sequential model. Each layer is treated as a single predictive coding layer, so if you want to include multiple functions within the same layer, you can wrap them in a separate call to sequential. For example the following code:
```
model=model=nn.Sequential(    
    nn.Conv2d(1,10,3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(10,10,3),
    nn.ReLU()
)
```
will treat each item as its own layer (5 layers in all). If you want to treat each "convolutional block" as a separate layer, you can instead do:
```
model=nn.Sequential(
    
    nn.Sequential(
      nn.Conv2d(1,10,3),
      nn.ReLU(),
      nn.MaxPool2d(2)
    ),
    
    nn.Sequential(
      nn.Conv2d(10,10,3),
      nn.ReLU()
    )
)
```
which has just 2 layers.

###  `LossFun`

`LossFun` is the loss function to be used, for example `LossFun = nn.MSELoss()`

###  `X` and `Y`
`X` and `Y` are batches or mini-batches of inputs and labels.

###  `ErrType`

`ErrType` is a string that specifies which algorithm to use for computing prediction errors and beliefs. It should be equal to `'Strict'`, `'FixedPred'`, or `'Exact'`. `'Strict'` uses a strict interpretation of predictive coding (without the fixed prediction assumption), `'FixedPred'` uses the fixed prediction assumption, and `'Exact'` computes the exact gradients (same as those computed by backpropagation). See "On the relationship between predictive coding and backpropagation" for more information on these algorithms.

###  `eta` and `n`

`eta` and `n` are the step size and number of steps to use for the iterations that compute the prediction errors and beliefs. These parameters are not used when `ErrType='Exact'`

### `vinit`

`vinit` is the initial value of the beliefs, which is only used when `ErrType='Strict'` If no initial belief is passed in, then the default value is the result from a forward pass, `vinit=vhat` 

## Outputs from `PCInfer`: 

###  `vhat`

`vhat` is a list of activations from a feedforward pass through the network (one item in the list for each layer). So `vhat[-1]` is the network's output (equivalent to `model(X)`)

### `Loss` and `dLdy`

`Loss` is the loss on the inputs, `X`. Equivalent to `LossFun(model(X), Y)` and `dLdy` is the gradient of the loss with respect to the network's output (`vhat[-1]`)

###  `v` and `epsilon`

`v` is a list of beliefs and `epsilon` a list of prediction errors. The algorithm used to compute these values depends on `ErrType`. For `ErrType='Exact'`, for example, `epsilon` contains the gradient of the loss with respect to each layers' activation (so `epsilon[-1]==dLdy` for example) and `v` is equal to `vhat-epsilon`. For other values of `ErrType`, refer to the paper for an explanation of how `v` and `epsilon` are computed.

## Side effect of `PCInfer` on `model`

In addition to computing its outputs, `PCInfer` modifies the `.grad` attributes of all parameters in `model` to the parameter update values computed by predictive coding. For `ErrType='Exact'`, the gradients are set to the gradient of the loss with respect to that parameter, i.e., the same values computed by calling `Loss.backward()` after a single forward pass. For other values of `ErrType`, refer to the paper for an explanation of how the parameter updates are computed.

## Example use of `PCInfer`

The following code block is equivalent to a standard forward and backward pass:
```
vhat,Loss,dLdy,v,epsilon=PCInfer(model,LossFun,X,Y,"Exact")
optimizer.step()
```
where where `optimizer` is an optimizer created using the PyTorch `optim` class, e.g., by calling `optimizer=optim.Adam(model.parameters())` before the call to `PCInfer`. To use predictive coding without the fixed prediction assumption, one would instead do
```
vhat,Loss,dLdy,v,epsilon=PCInfer(model,LossFun,X,Y,"Strict",eta,n)
optimizer.step()
```
and to use the fixed prediction assumption, simply replace `"Strict"` above with `"FixedPred"`.

### A complete example of training a convolutional neural network on MNIST is provided in the file `Example1.ipynb`
