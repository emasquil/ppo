# ppo
Proximal Policy Optimization Algorithm implementation for the Deep Reinforcement Learning course @MVA


## Install
First you need to clone the repository. For that, you can use the following command line:
```Bash
git clone git@github.com:emasquil/ppo.git
```
Then we recommand using a virtual environment, this can be done by the following:
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


## Contributing
Before any pull request, please make sure to format your code using the following:
```Bash
black -l 120 ./
```