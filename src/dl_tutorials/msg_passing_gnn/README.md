## Message Passing Graph Neural Network

This example provides recommendations using the pre-2020 message passing style graph neural networks.

The task is to predict an existing user's rating for a movie they haven't seen yet.

## References
Tutorial for the data
* [https://pytorch-geometric.readthedocs.io/en/latest/tutorial/load_csv.html]

It is based on the following

* [https://distill.pub/2021/gnn-intro/]

## Libraries

`pytorch-geometric` is the most active, general purpose graph neural network library. Unfortunately its documentation is hit or miss and the APIs somewhat inconsistent.

Other options

* https://www.dgl.ai/ - doesn't work on Mac
* https://pytorch-geometric-temporal.readthedocs.io/ - extension for dynamic graphs
* https://torchdrug.ai/ - GNNs specific to molecules and chemistry
