import torch
import torch.nn as nn
from torch.nn.init import normal_

from deepcarskit.model.context_recommender import ContextRecommender
from recbole.model.layers import MLPLayers
from recbole.utils import InputType, EvaluatorType
import numpy as np


class NeuCMF0iae3T(ContextRecommender):

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NeuCMF0iae3T, self).__init__(config, dataset)

        # load parameters info
        self.mf_embedding_size = config['mf_embedding_size']
        self.mlp_embedding_size = config['mlp_embedding_size']
        self.ae_embedding_size = 32
        self.ae_hidden_size = [128, 64]
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.mf_train = config['mf_train']
        self.mlp_train = config['mlp_train']
        self.ae_train = True # Default to True if not provided
        self.use_pretrain = config['use_pretrain']
        self.mf_pretrain_path = config['mf_pretrain_path']
        self.mlp_pretrain_path = config['mlp_pretrain_path']
        # define layers and loss
        self.user_mf_embedding = nn.Embedding(self.n_users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(self.n_items, self.mf_embedding_size)
        self.user_mlp_embedding = nn.Embedding(self.n_users, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(self.n_items, self.mlp_embedding_size)
        self.context_dimensions_mlp_embedding = []
        for i in range(0, self.n_contexts_dim):
            self.context_dimensions_mlp_embedding.append(nn.Embedding(self.n_contexts_conditions[i], self.mlp_embedding_size).to(self.device))

        # Autoencoder embeddings
        self.user_ae_embedding = nn.Embedding(self.n_users, self.ae_embedding_size)
        self.item_ae_embedding = nn.Embedding(self.n_items, self.ae_embedding_size)
        self.context_dimensions_ae_embedding = []
        for i in range(0, self.n_contexts_dim):
            self.context_dimensions_ae_embedding.append(nn.Embedding(self.n_contexts_conditions[i], self.ae_embedding_size).to(self.device))

        # mlp layers = user, item, context_situation
        self.mlp_layers = MLPLayers([(2 + self.n_contexts_dim) * self.mlp_embedding_size] + self.mlp_hidden_size, self.dropout_prob)
        self.mlp_layers.logger = None  # remove logger to use torch.save()

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

        if self.mf_train and self.mlp_train and self.ae_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size + self.mlp_hidden_size[-1] + self.ae_hidden_size[1], 1)
        elif self.mf_train and self.mlp_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size + self.mlp_hidden_size[-1], 1)
        elif self.mf_train and self.ae_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size + self.ae_hidden_size[1], 1)
        elif self.mlp_train and self.ae_train:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1] + self.ae_hidden_size[1], 1)
        elif self.mf_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size, 1)
        elif self.mlp_train:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
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
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)
        user_ae_e = self.user_ae_embedding(user)
        item_ae_e = self.item_ae_embedding(item)

        context_situation_mlp_e = None
        context_situation_ae_e = None
        for i in range(0, self.n_contexts_dim):
            condition = context_situation_list[i]
            embd_mlp = self.context_dimensions_mlp_embedding[i](condition)
            embd_ae = self.context_dimensions_ae_embedding[i](condition)
            if context_situation_mlp_e is None:
                context_situation_mlp_e = embd_mlp
                context_situation_ae_e = embd_ae
            else:
                context_situation_mlp_e = torch.cat((context_situation_mlp_e, embd_mlp), -1)
                context_situation_ae_e = torch.cat((context_situation_ae_e, embd_ae), -1)

        if self.mf_train:
            mf_output = torch.mul(user_mf_e, item_mf_e)  # [batch_size, embedding_size]
        if self.mlp_train:
            mlp_output = self.mlp_layers(torch.cat((user_mlp_e, item_mlp_e, context_situation_mlp_e), -1))  # [batch_size, layers[-1]]
        if self.ae_train:
            ae_input = torch.cat((user_ae_e, item_ae_e, context_situation_ae_e), -1)
            encoded = self.encoder(ae_input)
            #decoded = self.decoder(encoded)
            ae_output = encoded

        if self.mf_train and self.mlp_train and self.ae_train:
            output = self.actfun(self.predict_layer(torch.cat((mf_output, mlp_output, ae_output), -1)))
        elif self.mf_train and self.mlp_train:
            output = self.actfun(self.predict_layer(torch.cat((mf_output, mlp_output), -1)))
        elif self.mf_train and self.ae_train:
            output = self.actfun(self.predict_layer(torch.cat((mf_output, ae_output), -1)))
        elif self.mlp_train and self.ae_train:
            output = self.actfun(self.predict_layer(torch.cat((mlp_output, ae_output), -1)))
        elif self.mf_train:
            output = self.actfun(self.predict_layer(mf_output))
        elif self.mlp_train:
            output = self.actfun(self.predict_layer(mlp_output))
        elif self.ae_train:
            output = self.actfun(self.predict_layer(ae_output))
        else:
            raise RuntimeError('mf_train, mlp_train, and ae_train cannot all be False at the same time')
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
    

    def nonconformity_score(self, user, item, context_situation_list):
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
        
        ae_input = torch.cat((user_ae_e, item_ae_e, context_situation_ae_e), -1)
        encoded = self.encoder(ae_input)
        decoded = self.decoder(encoded)
        
        reconstruction_error = torch.nn.functional.mse_loss(ae_input, decoded, reduction='none').mean(dim=1)
        return reconstruction_error

    def calibrate(self, calibration_data):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            users, items, contexts, _ = calibration_data
            scores = self.nonconformity_score(users, items, contexts)
            self.calibration_scores = scores.cpu().detach().numpy()

    def conformal_predict(self, user, item, context_situation_list, alpha=0.1):
        self.calibrate()
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            #score = self.nonconformity_score(user, item, context_situation_list)
            prediction = self.forward(user, item, context_situation_list)
            
            # Compute the conformal prediction interval
            calibration_scores = np.sort(self.calibration_scores)
            n = len(calibration_scores)
            q = int(np.ceil((n + 1) * (1 - alpha)))
            threshold = calibration_scores[q - 1]
            
            lower = prediction - threshold
            upper = prediction + threshold
            
            return prediction, lower, upper

    def dump_parameters(self):
        r"""A simple implementation of dumping model parameters for pretrain.

        """
        if self.mf_train and not self.mlp_train and not self.ae_train:
            save_path = self.mf_pretrain_path
            torch.save(self, save_path)
        elif self.mlp_train and not self.mf_train and not self.ae_train:
            save_path = self.mlp_pretrain_path
            torch.save(self, save_path)
