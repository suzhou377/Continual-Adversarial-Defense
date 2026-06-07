import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
import torch
import numpy as np
import random
from torchvision import transforms
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
class dual_mlp(nn.Module):
    def __init__(self, input_size, output_size):
        super(dual_mlp, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
    def forward(self, x):
        out = self.fc1(x)
        return out   
'''cross attention'''
class ProtoCrossAttn(nn.Module):
    def __init__(self, D=512, n_head=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=D, num_heads=n_head, batch_first=True)
        self.q_proj = nn.Linear(D, D)
        self.kv_proj = nn.Linear(D, 2*D)
        self.w_param = nn.Parameter(torch.tensor(0.0))
    @property
    def w(self):
        return torch.sigmoid(self.w_param)
    def forward(self, emb, proto_pool):
        B, D = emb.shape
        residual = emb
        Q = self.q_proj(emb).unsqueeze(1)
        dim = proto_pool.shape[0]
        KV = self.kv_proj(proto_pool)
        K, V = KV.view(dim, 2, D).permute(1, 0, 2)
        K = K.expand(B, -1, -1)
        V = V.expand(B, -1, -1)
        out, _ = self.attn(Q, K, V)
        return (1 - self.w) * residual + self.w * out.squeeze(1)  # [B, D]
'''confidence'''
class ConfidenceLabelLoss(nn.Module):
    def __init__(self, emb_dim=512, hidden=64, theta=0.1, temp=0.2, eps=1e-8, conf_eps=1e-4):
        super().__init__()
        self.confidence_mlp = self._build_confidence_net(emb_dim, hidden, conf_eps)
        self.theta = theta  
        self.eps = eps 
        self.conf_eps = conf_eps
        self._log_temp = nn.Parameter(torch.tensor(float(temp)).log())
        self.w_param = nn.Parameter(torch.tensor(0.0)) 
    @property
    def temp(self):
        return self._log_temp.exp()
    @property
    def w(self):
        return torch.sigmoid(self.w_param)
    def _build_confidence_net(self, emb_dim, hidden):
        return nn.Sequential(
            nn.BatchNorm1d(emb_dim),
            nn.Linear(emb_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1))
    def compute_confidence(self, emb):
        conf = torch.sigmoid(self.confidence_mlp(emb))  
        conf = torch.clamp(conf, self.conf_eps, 1. - self.conf_eps) 
        conf_loss = -torch.log(conf).mean() 
        return conf, conf_loss
    def compute_entropy_loss(self, pred_dist):
        pred_detach = pred_dist.detach()
        log_pred = torch.log(pred_detach + self.eps)
        entropy = -(pred_detach * log_pred).sum(dim=1).mean()
        return entropy
    def compute_label_loss(self, logit, confidence, labels_onehot, criterion, labels):
        pred_original = F.softmax(logit / self.temp, dim=-1)
        pred_original = torch.clamp(pred_original, self.eps, 1. - self.eps)
        b = Bernoulli(confidence.data.new_empty(confidence.size()).uniform_(0., 1.)).sample()
        conf_weight = confidence * b + (1 - b)
        pred_new = pred_original * conf_weight + labels_onehot * (1 - conf_weight)
        label_loss = criterion(torch.log(pred_new + self.eps), labels)
        return pred_new, label_loss
    def forward(self, emb, logit, labels, labels_onehot, criterion):
        confidence, conf_loss = self.compute_confidence(emb)
        pred_new, label_loss = self.compute_label_loss(logit=logit, confidence=confidence, labels_onehot=labels_onehot, criterion=criterion, labels=labels)
        entropy_loss = self.compute_entropy_loss(pred_new)
        total_loss = label_loss + self.theta * (self.w * conf_loss + (1 - self.w) * entropy_loss)
        return total_loss

class PrototypeSupConLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super(PrototypeSupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, prototypes, labels):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        norm_features = features / torch.norm(features, p=2, dim=1, keepdim=True)
        norm_prototypes = prototypes / torch.norm(prototypes, p=2, dim=1, keepdim=True)

        num_classes = prototypes.shape[0]

        feature_dot_prototype = torch.div(
            torch.matmul(norm_features, norm_prototypes.T),
            self.temperature
        )

        logits_max, _ = torch.max(feature_dot_prototype, dim=1, keepdim=True)
        logits = feature_dot_prototype - logits_max.detach()

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, torch.arange(num_classes).to(device)).float()

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        condition = (mask_pos_pairs < 1e-6).to(device)
        mask_pos_pairs = torch.where(condition, torch.tensor(1, dtype=mask_pos_pairs.dtype, device=device), mask_pos_pairs)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - mean_log_prob_pos.mean()

        return loss

class EMAModel(torch.nn.Module):
    def __init__(self, model, ema_model, update_bn=True):
        super(EMAModel, self).__init__()
        self.model = model
        self.ema_model = ema_model
        self.update_bn = update_bn
        self.decay_rate = 0.
    def forward(self, x):
        x = self.ema_model(x)
        return x
    def update(self, epoch, ema_epoch, decay):
        if epoch < ema_epoch:
            self.decay_rate = 0.
        else:
            self.decay_rate = decay
        with torch.no_grad():
            for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_param.data.mul_(self.decay_rate).add_(param.data, alpha=1 - self.decay_rate)
            if self.update_bn:
                for module, ema_module in zip(self.model.modules(), self.ema_model.modules()):
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        ema_module.running_mean.mul_(self.decay_rate).add_(module.running_mean,
                                                                           alpha=1 - self.decay_rate)
                        ema_module.running_var.mul_(self.decay_rate).add_(module.running_var, alpha=1 - self.decay_rate)
                        ema_module.num_batches_tracked = module.num_batches_tracked

def calculate_prototype(logit, cluster, unique_label, y):
    prototypes = torch.zeros(len(unique_label), logit.size(1)).to(logit.device)
    for index in unique_label:
        selected = torch.where(cluster == index)[0]
        selected_logits = torch.index_select(logit, 0, selected)
        if selected_logits.shape[0] == 0:
            selected_indices = torch.where(y == index)[0]
            if selected_indices.numel() == 0:
                selected_logits = logit[torch.randint(len(logit), (1,))]
            else:
                random_index = torch.randint(selected_indices.numel(), (1,)).item()
                selected_logits = logit[selected_indices[random_index].unsqueeze(0)]
        prototypes[index] = selected_logits.mean(dim=0)
    return prototypes