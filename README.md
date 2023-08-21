# dist_curve
> Fast Nonparametric Estimation of Class Proportions in the Positive Unlabeled Classification Setting


## Install

`conda create -n dist_curve_env python=3.9`

`conda activate dist_curve_env`

`git clone git@github.com:Dzeiberg/dist_curve.git`

`python -m pip install -e dist_curve`

## Make Curve

```python
import numpy as np
from dist_curve.curve_constructor import makeCurve, plotCurve

alpha = 0.4
posSize = 100
mixSize = 500
dim = 1
posInstances = np.random.normal(loc=1,scale=1,size=(posSize, dim))

mixInstances = np.concatenate((np.random.normal(loc=1, scale=1, size=(int(mixSize*(alpha)), dim)),
                               np.random.normal(loc=3,scale=1,size=(int(mixSize * (1-alpha)), dim))),
                              axis=0)

curve = makeCurve(posInstances, mixInstances,)

plotCurve(curve)
```

# Estimate Class Prior

[Download Model](https://zenodo.org/record/8269226/files/model.hdf5)
```shell
$: wget https://zenodo.org/record/8269226/files/model.hdf5
```
```python
from dist_curve.model import getTrainedEstimator
```

```python
pathToModel = "./model.hdf5"
```

```python
model = getTrainedEstimator(pathToModel)
```

```python
model.predict(curve.reshape((1,-1))/curve.sum())
```
