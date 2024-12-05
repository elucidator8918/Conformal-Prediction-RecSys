class ConformalNeuCMF0iae3T(NeuCMF0iae3T):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.calibration_scores = None

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
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            score = self.nonconformity_score(user, item, context_situation_list)
            prediction = self.forward(user, item, context_situation_list)
            
            # Compute the conformal prediction interval
            calibration_scores = np.sort(self.calibration_scores)
            n = len(calibration_scores)
            q = int(np.ceil((n + 1) * (1 - alpha)))
            threshold = calibration_scores[q - 1]
            
            lower = prediction - threshold
            upper = prediction + threshold
            
            return prediction, lower, upper
