#!/usr/bin/env python3
""""
Lightning trainer for the graph neural network that predicts movie rankings.
"""
import einops
import torch
import lightning.pytorch as pl


class GnnBlock(torch.nn.Module):
    """This block takes the attached nodes and edges from the previous layer.
    Using the same dimension for the movie nodes and the user node embeddings.
    Not sure how to do other options
    """
    def __init__(self, embd_dim: int):
        self.lin_a = torch.nn.Linear(embd_dim + 1, embd_dim)
        self.lin_b = torch.nn.Linear(embd_dim, embd_dim)
        self.act = torch.nn.ReLU()

    def forward(self, node_embd: torch.Tensor,
                edge_ratings: torch.Tensor,
                neighbor_embd: torch.Tensor) -> torch.Tensor:
        """
        Compute updated embeddings for one node. This is brute force since I
        don't yet know good ways to represent a graph.
        :param node_embd: (embd_dim,) The existing embedding for this node
        :param edge_ratings: (n_edges,) User<->movie ratings on the edges.
        :param neighbor_embd: (n_edges, embd_dim)
        """
        # (n_edges, embd_dim + 1)
        x = torch.stack(
            edge_ratings.reshape((-1, 1)),
            neighbor_embd,
            dim=1)
        x = self.lin_a(x)  # (n_edges, embd_dim)
        x = self.act(x) + node_embd  # (n_edges, embd_dim)
        assert x.shape == neighbor_embd.shape
        x = self.lin_b(x)
        x = self.act(x)

        return x
