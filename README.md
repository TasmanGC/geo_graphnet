# geo_graphnet
This project applyies spatial graph construction, and Graph Neural Networks to geophysical interpretation problems. Some of the limitations of simpler neural network implementations, can be overcome by using a graph structure to capture the implicit relationships between points in a dataset. These relationships can be critical to domain specific classification tasks. 

This project is ongoing with the intent of extending this repositories functionality to cover the technical implementation of the two technical papers currently undergoing review.



### Usage
You can run a one shot experiment using configs defined in the config dir and any node/edge set you have access to.

    >>>(.venv) C:\path\to\repo\geo_graphnet> python run.py 

Alternatively you can complete an entire study by modifying the values in the optuna_hyper.py and running the below. This will complete an entire study, and save a {e_name}.db file as well.

    >>>(.venv) C:\path\to\repo\geo_graphnet> python optuna_hyper.py 

To consider the hyperparameter tuning space of that optuna study you can consider using the Optuna Dashboard.

    >>>(.venv) C:\path\to\repo\geo_graphnet> optuna-dashboard sqlite:///out/{e_name}/{e_name}.db

**NOTE:** Legacy functionality is still being ported over so some models while implemented are untested. Further given the stochastic nature of ML experiments repeatibility given the same configs is not guaranteed.
### Key Requirements
The default requirements assume windows 64 bit architecture, and pytorch with no CUDA support. The full requirements are caputured in the requirments.txt we used pip and venv for environemental management. 

- Python  3.10
- DGL 1.1.1
- torch 2.0.1
- Optuna (optional)

# Module Summary
The module builds around three core dependencies to implement the training/prediction process in a scalable and semi-autonomous manner. As it stands only the semi-supervised methodology has been ported over from a private repo. But future releases will include a supervised workflow.

In the diagram below the project is highlighted in blue, where key classes and methods from other libries are [Deep Graph Library](https://www.dgl.ai/)(orange) and [Optuna](https://optuna.org/) (green). Both are amazing projects and I would highly recomend checking them out. 

Using Deep Graph Library for the Graph Object and Graph Neural Net Implementation, while also using Optuna's study class to efficiently implent the hyperparameter tuning process. 

![An Evolving map of the module.](./docs/Module_Structure.drawio.svg)
