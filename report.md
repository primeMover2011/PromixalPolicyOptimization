# Project report

## Learning algorithm


## Parameters and hyperparameters

### Neural network architecture

The network consists of two networks: The actor network and the critic network.

*The actor network* takes a state tensor as an input and outputs an action.

*The critic network* is not directly needed for the PPO algorithm (original paper describes policy network and surrogate function which counts ration of new action probabilites to old ones - actor would suffice) but it's very helpful to compute advantages which requires value for state.

#### Actor network

- 3 fully connected layers
- 33 input nodes: _size of state vector_
- 4 output nodes: _size of action vector_
- 256 hidden nodes in each layer
- ReLU activations, tanh on last layer

#### Critic network

- 3 fully connected layers
- 33 input nodes [observation vector size], 1 output nodes, 512 hidden nodes in each layer
- ReLU activations, no activation on last layer

### Main hyperparameters

- Discount rate - `0.99`
- Tau - `0.95`
- Rollout length - `2048`
- Optimization epochs - `10`
- Gradient clip - `0.2`
- Learning rate - `3e-4`

## Results


## Next steps

