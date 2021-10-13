## Embedding Imputation using Personalized Propagation with Neural Predictions ##


### Requirements: ###

torch


### Usage: ###

cd tasks/class\_fin

python class\_lsi.py google aff 8 42

python train.py


cd tasks/regression

python imputation.py google glove 8 42 1 200

python train.py


### References: ###

[Latent Semantic Imputation](https://arxiv.org/pdf/1905.08900.pdf)

[APPNP](https://arxiv.org/pdf/1810.05997.pdf)
