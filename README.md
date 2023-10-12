# BRNet

This repository contains the code for performing model training along with various models trained on the ICSD dataset using Elemental Fraction (EF) as the model input.

## Installation Requirements

The basic requirement for using the files is a Python 3.6.3 environment with the packages listed in requirements.txt. It is advisable to create a virtual environment with the correct dependencies.

The work-related experiments were performed on Linux Fedora 7.9 Maipo. The code should be able to work on other Operating Systems as well, but it has not been tested elsewhere.

## Source Files

Here is a brief description of the folder content:

* [`automl`](./automl): code for training the automl model from scratch used for benchmarking.

* [`brnet`](./brnet): code for training the BRNet model from scratch.

* [`dataset`](./dataset): link to download different datasets used for training the BRNet model.

* [`model`](./model): various models trained on the ICSD dataset using Elemental Fraction (EF) as the model input and the BRNet model. 

## Running the code

Run the command below (after defining the data path, where the dataset contains the compound/formula in one of the columns named "pretty_comp" in the column to perform featurization), which uses BRNet architecture to perform model training:

`python BRNet.py -es 100 -sm "model_elemnet_stability" -prop "property"`

Where `-es` defines the number of epochs for early stopping, `-sm` defines the name of the saved model, and `-prop` defines the name of the column that is used for target materials property to train the model. Users can also modify the code to take custom features as input by modifying the code.

## Developer Team

The code was developed by Vishu Gupta from the <a href="http://cucis.ece.northwestern.edu/">CUCIS</a> group at the Electrical and Computer Engineering Department at Northwestern University.

## Publication

1. Vishu Gupta, Wei-keng Liao, Alok Choudhary, and Ankit Agrawal. "Brnet: Branched residual network for fast and accurate predictive modeling of materials properties." In Proceedings of the 2022 SIAM international conference on data mining (SDM), pp. 343-351. Society for Industrial and Applied Mathematics, 2022. [<a href="https://epubs.siam.org/doi/abs/10.1137/1.9781611977172.39">DOI</a>] [<a href="https://epubs.siam.org/doi/epdf/10.1137/1.9781611977172.39">PDF</a>]

```tex
@inproceedings{gupta2022brnet,
  title={Brnet: Branched residual network for fast and accurate predictive modeling of materials properties},
  author={Gupta, Vishu and Liao, Wei-keng and Choudhary, Alok and Agrawal, Ankit},
  booktitle={Proceedings of the 2022 SIAM international conference on data mining (SDM)},
  pages={343--351},
  year={2022},
  organization={SIAM}
}
```

## Disclaimer

The research code shared in this repository is shared without any support or guarantee of its quality. However, please do raise an issue if you find anything wrong, and I will try my best to address it.

email: vishugupta2020@u.northwestern.edu

Copyright (C) 2023, Northwestern University.

See COPYRIGHT notice in the top-level directory.

## Funding Support

This work was performed under the following financial assistance award 70NANB19H005 from the U.S. Department of Commerce, National Institute of Standards and Technology, as part of the Center for Hierarchical Materials Design (CHiMaD). Partial support is also acknowledged from DOE awards DE-SC0014330, DE-SC0019358, and DE-SC0021399.
