import torch
from torch import nn
class MF(nn.Module):
    def __init__(self, model, n_users, n_items, n_layers, n_factors=16, droprate=0, MF_model=None, MLP_model=None):
        super(MF, self).__init__()
        self.model = model
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors

        self.MF_model = MF_model
        self.MLP_model = MLP_model

        self.user_embeddings_mf = nn.Embedding(n_users, n_factors)
        self.item_embeddings_mf = nn.Embedding(n_items, n_factors)

        self.user_embeddings_mlp = nn.Embedding(n_users, n_factors)
        self.item_embeddings_mlp = nn.Embedding(n_items, n_factors)

        self.droprate = droprate
        mlp_layers = []
        mlp_layers.append(nn.Linear(n_factors * 2, n_factors * (2 ** n_layers)))
        mlp_layers.append(nn.ReLU(inplace=True))
        for i in range(n_layers):
            input_dim = n_factors * (2 ** (n_layers - i))
            mlp_layers.append(nn.Linear(input_dim, input_dim // 2))
            mlp_layers.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp_layers)
        
        if self.model in ['MLP', 'MF']:
            predict_size = n_factors
        else:
            predict_size = n_factors * 2

        self.output = nn.Linear(predict_size, 1)
          
        self._init_weight_()
    
    def _init_weight_(self):
        if self.model == 'NMF-pre':
            self.user_embeddings_mf.weight.data.copy_(
                self.MF_model.user_embeddings_mf.weight)
            self.item_embeddings_mf.weight.data.copy_(
                self.MF_model.item_embeddings_mf.weight)
            self.user_embeddings_mlp.weight.data.copy_(
                self.MLP_model.user_embeddings_mlp.weight)
            self.item_embeddings_mlp.weight.data.copy_(
                self.MLP_model.item_embeddings_mlp.weight)

            for m1, m2 in zip(self.mlp, self.MLP_model.mlp):
                if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
                    m1.weight.data.copy_(m2.weight)
                    m1.bias.data.copy_(m2.bias)

            self.output.weight.data.copy_(0.5 * torch.cat([
                self.MF_model.output.weight, self.MLP_model.output.weight], dim=1))
            self.output.bias.data.copy_(0.5 * (
                self.MF_model.output.bias + self.MLP_model.output.bias))    

        else:
            nn.init.normal_(self.user_embeddings_mf.weight, std=0.01)
            nn.init.normal_(self.item_embeddings_mf.weight, std=0.01)
            nn.init.normal_(self.user_embeddings_mlp.weight, std=0.01)
            nn.init.normal_(self.item_embeddings_mlp.weight, std=0.01)

            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    m.bias.data.zero_()

            nn.init.kaiming_uniform_(self.output.weight, a=1, nonlinearity='sigmoid')
            self.output.bias.data.zero_()

    def forward(self, idx_users, idx_items):
        if not self.model == 'MLP':
            user_embs_mf = self.user_embeddings_mf(idx_users)
            item_embs_mf = self.item_embeddings_mf(idx_items)
            mf_output = user_embs_mf * item_embs_mf
        if not self.model == 'MF':
            user_embs_mlp = self.user_embeddings_mlp(idx_users)
            item_embs_mlp = self.item_embeddings_mlp(idx_items)

            interaction = torch.cat([user_embs_mlp, item_embs_mlp], dim=-1)
            mlp_output = self.mlp(interaction)
        
        if self.model == 'MF':
            concat = mf_output
        elif self.model == 'MLP':
            concat = mlp_output
        else:
            concat = torch.cat((mf_output, mlp_output), -1)
        
        return self.output(concat).squeeze()

        



    

    

