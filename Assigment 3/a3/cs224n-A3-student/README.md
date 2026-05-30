# cs224n-A3


## Install 

First, create a new conda environment with Python 3.10:

```bash
conda create -n cs224n-A3 python=3.10
```
Activate the environment:

```bash
conda activate cs224n-A3
```

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

## Running Tests

To run tests (note you need to be in the `tests/` directory):

```bash 
cd tests
pytest
```

If you want to run a specific test, e.g. `test_forward`:

```bash
pytest test_student.py::test_forward
```
