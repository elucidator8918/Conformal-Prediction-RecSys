import torch
import torch.nn as nn
from torch.nn.init import normal_

from deepcarskit.model.context_recommender import ContextRecommender
from recbole.utils import InputType


class NeuCMF0iae(ContextRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NeuCMF0iae, self).__init__(config, dataset)

        # load parameters info
        self.mf_embedding_size = config['mf_embedding_size']
        self.ae_embedding_size = 32
        self.ae_hidden_size = [128, 64]
        self.dropout_prob = config['dropout_prob']
        self.mf_train = config['mf_train']
        self.ae_train = True # Default to True if not provided
        self.use_pretrain = config['use_pretrain']
        self.mf_pretrain_path = config['mf_pretrain_path']
        self.mlp_pretrain_path = config['mlp_pretrain_path']
        # define layers and loss
        self.user_mf_embedding = nn.Embedding(self.n_users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(self.n_items, self.mf_embedding_size)

        # Autoencoder embeddings
        self.user_ae_embedding = nn.Embedding(self.n_users, self.ae_embedding_size)
        self.item_ae_embedding = nn.Embedding(self.n_items, self.ae_embedding_size)
        self.context_dimensions_ae_embedding = []
        for i in range(0, self.n_contexts_dim):
            self.context_dimensions_ae_embedding.append(nn.Embedding(self.n_contexts_conditions[i], self.ae_embedding_size).to(self.device))

        # autoencoder layers
        self.encoder = nn.Sequential(
            nn.Linear((2 + self.n_contexts_dim) * self.ae_embedding_size, self.ae_hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.ae_hidden_size[0], self.ae_hidden_size[1])
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.ae_hidden_size[1], self.ae_hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.ae_hidden_size[0], (2 + self.n_contexts_dim) * self.ae_embedding_size)
        )

        if self.mf_train and self.ae_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size + self.ae_hidden_size[1], 1)
        elif self.mf_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size, 1)
        elif self.ae_train:
            self.predict_layer = nn.Linear(self.ae_hidden_size[1], 1)

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
        user_ae_e = self.user_ae_embedding(user)
        item_ae_e = self.item_ae_embedding(item)

        context_situation_ae_e = None
        for i in range(0, self.n_contexts_dim):
            condition = context_situation_list[i]
            embd_ae = self.context_dimensions_ae_embedding[i](condition)
            if context_situation_ae_e is None:
                context_situation_ae_e = embd_ae
            else:
                context_situation_ae_e = torch.cat((context_situation_ae_e, embd_ae), -1)

        if self.mf_train:
            mf_output = torch.mul(user_mf_e, item_mf_e)  # [batch_size, embedding_size]
        if self.ae_train:
            ae_input = torch.cat((user_ae_e, item_ae_e, context_situation_ae_e), -1)
            encoded = self.encoder(ae_input)
            decoded = self.decoder(encoded)
            ae_output = encoded

        if self.mf_train and self.ae_train:
            output = self.actfun(self.predict_layer(torch.cat((mf_output, ae_output), -1)))
        elif self.mf_train:
            output = self.actfun(self.predict_layer(mf_output))
        elif self.ae_train:
            output = self.actfun(self.predict_layer(ae_output))
        else:
            raise RuntimeError('mf_train and ae_train cannot both be False at the same time')
        return output.squeeze(-1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        context_situation_list = self.getContextSituationList(interaction, self.CONTEXTS)
        label = interaction[self.LABEL]

        output = self.forward(user, item, context_situation_list)
        return self.loss(output, label)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        context_situation_list = self.getContextSituationList(interaction, self.CONTEXTS)
        return self.forward(user, item, context_situation_list)

    def dump_parameters(self):
        r"""A simple implementation of dumping model parameters for pretrain.

        """
        if self.mf_train and not self.ae_train:
            save_path = self.mf_pretrain_path
            torch.save(self, save_path)
