import torch
import torch.nn as nn
import higher

from networks import Conv4

class MAML(nn.Module):

    def __init__(self, num_ways, input_size, T=1, second_order=False, inner_lr=0.4, **kwargs):
        super().__init__()
        self.num_ways = num_ways
        self.input_size = input_size
        self.num_updates = T
        self.second_order = second_order
        self.inner_loss = nn.CrossEntropyLoss()
        self.inner_lr = inner_lr

        self.network = Conv4(self.num_ways, img_size=int(input_size**0.5)) 


    # controller input = image + label_previous
    def apply(self, x_supp, y_supp, x_query, y_query, training=False):
        """
        Perform the inner-level learning procedure of MAML: adapt to the given task 
        using the support set. It returns the predictions on the query set, as well as the loss
        on the query set (cross-entropy).
        You may want to set the gradients manually for the base-learner parameters 

        :param x_supp (torch.Tensor): the support input images of shape (num_support_examples, num channels, img width, img height)
        :param y_supp (torch.Tensor): the support ground-truth labels
        :param x_query (torch.Tensor): the query inputs images of shape (num_query_inputs, num channels, img width, img height)
        :param y_query (torch.Tensor): the query ground-truth labels

        :returns:
          - query_preds (torch.Tensor): the predictions of the query inputs
          - query_loss (torch.Tensor): the cross-entropy loss on the query inputs
        """
        
        fast_weights = [p.clone() for p in self.network.parameters()]

        for k in range (self.num_updates):
            preds = self.network(x_supp, weights=fast_weights)
            loss = self.inner_loss(preds, y_supp)
            grad = torch.autograd.grad(loss, fast_weights, create_graph=self.second_order)
            fast_weights = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, fast_weights)))
        
        query_preds = self.network(x_query, weights=fast_weights)
        query_loss = self.inner_loss(query_preds, y_query)

        if training:
            query_loss.backward() # do not remove this if statement, otherwise it won't train

        return query_preds, query_loss
