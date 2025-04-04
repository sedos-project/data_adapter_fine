# SEDOS Data Adapter ETHOS.FINE
A data adapter to translate sedos datasets to the ETHOS.FINE format and build an energy system model.

## Get started

### Installation
 - create environment with conda/ mamba using the `environment.yml` file
 - install the package using `pip install -e . --no-deps` in the root folder of the repository

## data_adapter_fine folder
This folder contains the main code of the data adapter. It is structured as follows:
 - dataAdapter.py: contains the main code of the data adapter
 - constructor.py: contains the code to build the components of the energy system model
 - utils.py: contains utility functions used in the data adapter
 - postprocess.py: contains the code to post-process the data and create figures

## collections folder
This folder contains the data package used to build the energy system model. New data from the OEP will be saved here.

## examples folder
This folder contains examples of how to use the data adapter. It is structured as follows:
 - example.py: contains an example of how to use the data adapter

# structures folder
This folder contains xlsx files for model structures. New structure files need to be saved here.
