
import torch 

print('Running TorchSeq2PC.py')

#NEWER

# Perform a forward pass on a Sequential model
# where X,Y are one batch of inputs,labels
# Returns activations for all layers (vhat), loss, and gradient of loss
# wrt last-layer activations (dLdy)
# vhat,Loss,dLdy=FwdPassPlus(model,LossFun,X,Y)
def FwdPassPlus(model,LossFun,X,Y):
    
    # Number of layers, counting the input as layer 0 
    DepthPlusOne=len(model)+1 

    # Forward pass
    vhat=[None]*DepthPlusOne
    vhat[0]=X
    for layer in range(1,DepthPlusOne):
      f=model[layer-1]        
      vhat[layer]=f(vhat[layer-1])
    Loss = LossFun(vhat[-1], Y)

    # Compute gradient of loss with respect to output
    dLdy=torch.autograd.grad(Loss, vhat[-1])[0]

    return vhat,Loss,dLdy



# Compute prediction errors (epsilon) using 
# predictive coding algorithm modified by
# the fixed prediction assumption
# see Millidge et al. 
# v,epsilon=ModifiedPCPredErrs(model,vhat,dLdy,eta=1,n=None)
def ModifiedPCPredErrs(model,vhat,dLdy,eta=1,n=None):

    # Number of layers, counting the input as layer 0 
    DepthPlusOne=len(model)+1 

    if n==None:
      n=len(model)

    # Initialize epsilons
    epsilon=[None]*DepthPlusOne
    epsilon[-1]=dLdy

    # Initialize v to a copy of vhat with no gradients needed
    # (can this be moved up to the loop above?)
    v=[None]*DepthPlusOne
    for layer in range(DepthPlusOne):
      v[layer]=vhat[layer].clone().detach()
    
    # Iterative updates of v and epsilon using stored values of vhat    
    for i in range(n):
      for layer in reversed(range(DepthPlusOne-1)):#range(DepthPlusOne-2,-1,-1):  
        epsilon[layer]=vhat[layer]-v[layer]
        _,epsdfdv=torch.autograd.functional.vjp(model[layer],vhat[layer],epsilon[layer+1])               
        dv=epsilon[layer]-epsdfdv
        v[layer]=v[layer]+eta*dv
      
    return v,epsilon




# Compute prediction errors (epsilon) using 
# a strict interpretation of predictive coding
# without the fixed prediction assumption.
# v,epsilon=StrictPCPredErrs(model,vinit,LossFun,Y,eta,n)
def StrictPCPredErrs(model,vinit,LossFun,Y,eta,n):

    with torch.no_grad():
      # Number of layers, counting the input as layer 0 
      DepthPlusOne=len(model)+1 

      # Initialize epsilons
      epsilon=[None]*DepthPlusOne

      # Initialize v to a copy of vinit with no gradients needed
      # (can this be moved up to the loop above?)
      v=[None]*DepthPlusOne
      for layer in range(DepthPlusOne):
        v[layer]=vinit[layer].clone()    

    # Iterative updates of v and epsilon 
    for i in range(n):
      model.zero_grad()
      layer=DepthPlusOne-1
      vtilde=model[layer-1](v[layer-1])
      Loss=LossFun(vtilde,Y)
      epsilon[layer]=torch.autograd.grad(Loss,vtilde,retain_graph=False)[0] # -2 ~ DepthPlusOne-2
      for layer in reversed(range(1,DepthPlusOne-1)):
        epsilon[layer]=v[layer]-model[layer-1](v[layer-1])
        _,epsdfdv=torch.autograd.functional.vjp(model[layer],v[layer],epsilon[layer+1])               
        dv=-epsilon[layer]+epsdfdv
        v[layer]=v[layer]+eta*dv

    return v,epsilon


# Compute exact prediction errors (epsilon)
# define as the gradient of the loss wrt to
# the activations
# v,epsilon=ExactPredErrs(model,LossFun,X,Y,vhat=None)
def ExactPredErrs(model,LossFun,X,Y,vhat=None):

    # Number of layers, counting the input as layer 0 
    DepthPlusOne=len(model)+1

    # Forward pass if it wasn't passed in
    if vhat==None:
      vhat=[None]*DepthPlusOne
      vhat[0]=X
      for layer in range(1,DepthPlusOne):
        f=model[layer-1]
        vhat[layer]=f(vhat[layer-1])

    Loss = LossFun(vhat[-1], Y)
    
    epsilon=[None]*DepthPlusOne
    v=[None]*DepthPlusOne

    for layer in range(1,DepthPlusOne):          
      epsilon[layer]=torch.autograd.grad(Loss,vhat[layer],allow_unused=True,retain_graph=True)[0]      
      v[layer]=vhat[layer]-epsilon[layer]
    
    return v,epsilon

    
# Set gradients of model params based on PC approximations
def SetPCGrads(model,epsilon,X,vhat=None):
    
    # Number of layers, counting the input as layer 0 
    DepthPlusOne=len(model)+1

    # Forward pass if it wasn't passed in
    if vhat==None:
      vhat=[None]*DepthPlusOne
      vhat[0]=X
      for layer in range(1,DepthPlusOne):
        f=model[layer-1]
        vhat[layer]=f(vhat[layer-1])

    # Compute new parameter values    
    for layer in range(0,DepthPlusOne-1):
      for p in model[layer].parameters():
        dtheta=torch.autograd.grad(vhat[layer+1],p,grad_outputs=epsilon[layer+1],allow_unused=True,retain_graph=True)[0]
        p.grad = dtheta


# Do a whole PC step
# vhat,Loss,dLdy,v,epsilon=OnePCStep(model,LossFun,X,Y,eta=1,n=None,PCErrType="Modified")
def OnePCStep(model,LossFun,X,Y,eta=1,n=None,PCErrType="Modified"):
  
  if n==None:
    n=len(model)

  # Fwd pass (plus return vhat and dLdy)
  vhat,Loss,dLdy=FwdPassPlus(model,LossFun,X,Y)

  # Get beliefs and prediction errors
  if PCErrType=="Modified":
    v,epsilon=ModifiedPCPredErrs(model,vhat,dLdy,eta,n)
  elif PCErrType=="Strict":
    v,epsilon=StrictPCPredErrs(model,vhat,LossFun,Y,eta,n)
  elif PCErrType=="Exact":
    v,epsilon=ExactPredErrs(model,LossFun,X,Y)

  # Set gradients in model
  SetPCGrads(model,epsilon,X,vhat)

  return vhat,Loss,dLdy,v,epsilon


