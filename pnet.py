import torch
import torch.nn as nn

from networks import Conv4

class PrototypicalNetwork(nn.Module):

    def __init__(self, num_ways, input_size, similarity="euclidean", **kwargs):
        # euclidean means that we use the negative squared Euclidean distance as kernel
        super().__init__()
        self.num_ways = num_ways
        self.input_size = input_size
        self.criterion = nn.CrossEntropyLoss()

        self.network = Conv4(self.num_ways, img_size=int(input_size**0.5)) 
        self.similarity = similarity


    def apply(self, x_supp, y_supp, x_query, y_query, training=False):
        """
        Function that applies the Prototypical network to a given task and returns the predictions 
        on the query set as well as the loss on the query set

        :param x_supp (torch.Tensor): support set images
        :param y_supp (torch.Tensor): support set labels
        :param x_query (torch.Tensor): query set images
        :param y_query (torch.Tensor): query set labels
        :param training (bool): whether we are in training mode

        :return:
          - query_preds (torch.Tensor): our predictions for all query inputs of shape [#query inputs, #classes]
          - query_loss (torch.Tensor): the cross-entropy loss between the query predictions and y_query
        """

        # TODO: imeplement this function

        if training:
            query_loss.backward() # do not remove this if statement, otherwise it won't train

        raise NotImplementedError()