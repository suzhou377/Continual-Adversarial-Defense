import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from kmeans import kmeans
from function import encode_onehot, calculate_prototype
def cad_loss(encoder,
             classifier,
             move_average,
             cross_fusion,
             confidence_mlp,
             task_mlp,
             x_natural,
             y,
             optimizer,
             prototype_contrastive,
             natural_proto_pool,
             robust_proto_pool,
             proto_momentum,
             unique_label,
             criterion_nll,
             criterion_kl,
             step_size=2/255,
             epsilon=8/255,
             theta1=1,
             theta2=1,
             rampup=0,
             epoch=0):
    encoder.eval()
    classifier.eval()
    if epoch <= 40:
        perturb_steps = 0
    elif epoch <= 80:
        perturb_steps = 1
        step_size = epsilon
    else:
        perturb_steps = 10
    if perturb_steps > 0:
        x_adv = x_natural.detach() + torch.FloatTensor(*x_natural.shape).uniform_(-epsilon, epsilon).to(x_natural.device)
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_pgd = F.cross_entropy(classifier(encoder(x_adv)), y)
            grad = torch.autograd.grad(loss_pgd, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = x_natural.detach() + torch.FloatTensor(*x_natural.shape).uniform_(-epsilon, epsilon).to(x_natural.device)
    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    encoder.train()
    classifier.train()
    optimizer.zero_grad()
    natural_emb = encoder(x_natural)
    natural_logit = classifier(natural_emb)
    robust_emb = encoder(x_adv)
    robust_logit = classifier(robust_emb)
    mlp_robust_logit = task_mlp(robust_emb)
    with torch.no_grad():
        move_natural_logit, move_natural_emb = move_average(x_natural)
        move_robust_logit, move_robust_emb = move_average(x_adv)
    labels_onehot = encode_onehot(y, n_classes=10)
    ce_loss = (rampup * F.cross_entropy(robust_logit, y) + rampup * F.cross_entropy(mlp_robust_logit, y) +
               (1 - rampup) * confidence_mlp(emb=robust_emb, logit=robust_logit, labels=y, labels_onehot=labels_onehot, criterion=criterion_nll) +
               (1 - rampup) * confidence_mlp(emb=robust_emb, logit=mlp_robust_logit, labels=y, labels_onehot=labels_onehot, criterion=criterion_nll)) / 2
    if epoch <= 40:
        natural_cluster, robust_cluster = move_natural_logit.max(dim=1)[1], move_robust_logit.max(dim=1)[1]
        natural_center = calculate_prototype(move_natural_emb.detach(), natural_cluster, unique_label, y)
        robust_center = calculate_prototype(move_robust_emb.detach(), robust_cluster, unique_label, y)
    else:
        move_natural_emb = cross_fusion(move_natural_emb.detach(), natural_proto_pool)
        natural_cluster, natural_center = kmeans(move_natural_emb.detach(), num_clusters=10, unique_label=unique_label, y=y, distance='cosine')
        move_robust_emb = cross_fusion(move_robust_emb.detach(), robust_proto_pool)
        robust_cluster, robust_center = kmeans(move_robust_emb.detach(), num_clusters=10, unique_label=unique_label, y=y, distance='cosine')
        cld_loss = (prototype_contrastive(natural_emb, natural_center, labels=natural_cluster) +
                    prototype_contrastive(robust_emb, robust_center, labels=robust_cluster)) / 2
    align_loss = criterion_kl(F.log_softmax(robust_logit, dim=1), F.softmax(move_natural_logit, dim=1))
    if epoch <= 40:
        loss = ce_loss
    else:
        loss = ce_loss + theta1 * cld_loss + theta2 * align_loss
    natural_accuracy = (torch.argmax(natural_logit, dim=1) == y).sum().item()
    robust_accuracy = (torch.argmax(robust_logit, dim=1) == y).sum().item()
    natural_proto_pool = proto_momentum * natural_proto_pool + (1 - proto_momentum) * natural_center
    robust_proto_pool = proto_momentum * robust_proto_pool + (1 - proto_momentum) * robust_center
    return loss, robust_accuracy, natural_accuracy, natural_proto_pool.detach(), robust_proto_pool.detach()
