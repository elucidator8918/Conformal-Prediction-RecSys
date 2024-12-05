import torch
import torch.nn as nn
from torch.nn.init import normal_

from deepcarskit.model.context_recommender import ContextRecommender
from recbole.utils import InputType


class NeuCMF0i(ContextRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NeuCMF0i, self).__init__(config, dataset)

        # load parameters info
        self.mf_embedding_size = config['mf_embedding_size']
        self.mlp_embedding_size = config['mlp_embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.mf_train = config['mf_train']
        self.mlp_train = config['mlp_train']
        self.use_pretrain = config['use_pretrain']
        self.mf_pretrain_path = config['mf_pretrain_path']
        self.mlp_pretrain_path = config['mlp_pretrain_path']

        # Make sure these attributes are initialized
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)

        # define layers and loss
        self.user_mf_embedding = nn.Embedding(self.n_users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(self.n_items, self.mf_embedding_size)

        if self.mf_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size * 2, 1)
        else:
            raise ValueError('MF training must be enabled.')

        # parameters initialization
        if self.use_pretrain:
            self.load_pretrain()
        else:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item, context_situation_list):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)

        mf_output = torch.cat((user_mf_e, item_mf_e), dim=-1)  # concatenate user and item embeddings

        output = self.predict_layer(mf_output)
        return output.squeeze(-1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        output = self.forward(user, item, None)  # context_situation_list is not used for MF
        return self.loss(output, label)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item, None)

    def dump_parameters(self):
        r"""A simple implementation of dumping model parameters for pretrain."""
        save_path = self.mf_pretrain_path
        torch.save(self.state_dict(), save_path)
