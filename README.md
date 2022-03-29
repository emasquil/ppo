# ppo
Proximal Policy Optimization Algorithm implementation for the Deep Reinforcement Learning course @MVA


[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emasquil/ppo/blob/main/ppo.ipynb)


## Install
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
sudo apt-get install -y xvfb ffmpeg freeglut3-dev libosmesa6-dev patchelf
```


## Contributing
Before any pull request, please make sure to format your code using the following:
```Bash
black -l 120 ./
```