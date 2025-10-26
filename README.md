# Todo

## Core

- [ ] Complete Adam optimizer implementation in `neural_net/optimizers/adam.py`
- [ ] Add state variables (m, v, t)
- [ ] Implement Adam update logic in `step()` method
- [ ] Implement `reset()` method for state variables

- [ ] Add L1 regularization in `neural_net/training/regularizers.py`
- L2 regularization is already implemented

- [ ] Implement Batch Normalization layer in `neural_net/layers/batchnorm.py`
- [ ] Create BatchNorm class extending BaseLayer
- [ ] Implement forward pass with normalization
- [ ] Implement backward pass with gradient computation

## Experiment Configuration

- [ ] Finalize real experiment configurations in `experiments/configs/configurations.py`
- Current M0-M4 configs are AI-generated examples

## Documentation

- [ ] Add docstrings to all public methods
- [ ] Create usage examples
- [ ] Document experiment results
