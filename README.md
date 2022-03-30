# Proximal Policy Optimization (PPO)
![python](https://img.shields.io/badge/python-3.7%20|%203.8%20|%203.9-blue)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emasquil/ppo/blob/main/ppo.ipynb)

This repository provides a clean code of the PPO algorithm using [JAX](https://github.com/google/jax) and [Haiku](https://github.com/deepmind/dm-haiku).

Interested readers can have a look to our [report](./report.pdf) that goes deeper into the details.

Feel free to checkout to:
* [Environments](#environments)
* [Agents](#agents)
* [Tricks](#tricks)
* [How to run it](#how-to-run-it)
* [Contributing](#contributing)
* [Inspirations](#inspirations)


## Environments
### Continuous
- [Pendulum-v1](https://www.gymlibrary.ml/pages/environments/classic_control/pendulum). A show case with details is available [here](https://github.com/emasquil/ppo/blob/main/examples/Pendulum-v1.ipynb).
- [Reacher-v2](https://www.gymlibrary.ml/pages/environments/mujoco/reacher).

## Agents
### Continous 
- [random_agent](https://github.com/emasquil/ppo/blob/logger_actions/ppo/agents/random_agent.py): a random agent to test your implementation.
- [vanilla_ppo](https://github.com/emasquil/ppo/blob/logger_actions/ppo/agents/vanilla_ppo.py): a classic version of PPO.


## Tricks
### Networks
- [x] Separated value and policy networks.
- [x] The standard deviation of the action comes from one parameter only and is independant of the observation. (for continuous action setting only)
- [x] Orthogonal initialization of the weights and constant initialization for the biases.
- [x] Activation function are `tanh`.

### Training
- [x] Linear annealing of the learning rate.
- [x] Learning with minibatches.

### Loss
- [x] Using Generalized Advantage Estimation (GAE).
- [x] Clipped ratio  
- [x] Minimum between ratio x GAE and clipped_ratio x GAE
- [x] Normalized advantage for the policy loss.
- [ ] Clipped value loss
- [ ] Entropy bonus
- [x] Clipped gradient norm

## How to run it
### Fast and easy
Just click on this colab link and have fun with the code:
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emasquil/ppo/blob/main/ppo.ipynb)

### Run it locally
First you need to clone the repository. For that, you can use the following command line:
```Bash
git clone git@github.com:emasquil/ppo.git
```
Then we recommend using a virtual environment, this can be done by the following:
```Bash
python3 -m venv env
source env/bin/activate
```
Finally, in order to install the package, you can simply run:
```Bash
pip install -e .
```
If you are planning on developing the package you will need to add `[dev]` at the end. This gives:
```Bash
pip install -e .[dev]
```

This package uses MuJoCo environments, please install it by following these [instructions](https://github.com/openai/mujoco-py/).

Note that you might need to install the following.

```
sudo apt-get install -y xvfb ffmpeg freeglut3-dev libosmesa6-dev patchelf libglew-dev
```

## Contributing
Before any pull request, please make sure to format your code using the following:
```Bash
black -l 120 ./
```

## Inspirations

[vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)\
[openai/baselines](https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/model.py)\
[DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/ppo)\
[openai/spinningup](https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/ppo/ppo.py)\
[Costa Huang's blogpost](https://costa.sh/blog-the-32-implementation-details-of-ppo.html)