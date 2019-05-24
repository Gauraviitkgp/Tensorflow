# Tensorflow

## How to create a virtual enviroment

https://www.tensorflow.org/install/pip

Create a new virtual environment by choosing a Python interpreter and making a ./venv directory to hold it:

```
	virtualenv --system-site-packages -p python3 ./venv 
```

Activate the virtual environment using a shell-specific command:

```
source ./venv/bin/activate  # sh, bash, ksh, or zsh
```

When virtualenv is active, your shell prompt is prefixed with (venv).

Install packages within a virtual environment without affecting the host system setup. Start by upgrading pip:

```
	pip install --upgrade pip

	pip list  # show packages installed within the virtual environment
```

And to exit virtualenv later:
```
	deactivate  # don't exit until you're done using TensorFlow
```