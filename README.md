# Mole

ChemoInformatics Code Base


![GitHub CI](https://github.com/the-mama-ai/machi-light/actions/workflows/python-package-conda.yml/badge.svg)


## Installation


#### Option 1:



```
pip install git+https://{YOUR_GITHUB_TOKEN}@github.com/the-mama-ai/machi-light.git

```

(note this installation option does not guarantee that all conda requirements of the project are met)


#### Option 2:

Clone the repository
```
git clone https://github.com/the-mama-ai/machi-light.git

cd machi-light
```

Create a new conda environment. Depending on your machine, choose either `requirements/environment.yml` or `requirements/environment_cuda11.yml` to create a new conda environement.

```
# CPU option
conda env create -f requirements/environment.yml

# CUDA option
conda env create -f requirements/environment_cuda11.yml
```

Install `machi-light` 

```
python setup.py install
```

## Tutorials

Add tutorials here
