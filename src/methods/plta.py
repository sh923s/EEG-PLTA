import torch
import torch.nn as nn
from copy import deepcopy

import torch.bin
from ..utils import split_up_model
from torch.amp import autocast, GradScaler


class MyMem:
    def __init__(self, num_classes, capacity, conf_threshold):
        self.capacity = capacity
        self.conf_threshold = conf_threshold
        self.num_classes = num_classes
        self.mem = []
        self.feats_mem = None
        self.pred_prob_mem = None
        self.class_mem = None
        self.age_mem = None
        self.lambda_age = 1
        self.lambda_conf = 1
        
    def __len__(self):
        return len(self.mem)
    
    def get_age_score(self, age):
        return torch.sigmoid(-age/self.capacity)
    
    def get_conf_score(self, conf):
        return torch.exp((conf-1)/self.conf_threshold)
    
    def get_score(self, age, conf):
        return self.get_conf_score(conf)*self.lambda_conf + self.get_age_score(age)*self.lambda_age
    
    def update_mem_age(self):
        if self.age_mem is None:
            return
        else:
            self.age_mem += 1
    
    def update(self, x, fea, prob, conf):
        for i in range(x.shape[0]):
            c = conf[i]
            xx = x[i]
            if len(self.mem) < self.capacity:
                self.add_to_mem(xx, fea[i], prob[i])
            else:
                conf_mem = self.pred_prob_mem.max(1)[0]
                # get the score for each sample in the mem
                score_mem = self.get_score(self.age_mem.to(conf_mem.device), conf_mem)
                # find the most occupant class
                class_occupance = torch.bincount(self.class_mem)
                prevalent_class = class_occupance.argmax()
                prevalenet_samp_idx = torch.where(self.class_mem == prevalent_class)[0]
                score_mem_prevalent = score_mem[prevalenet_samp_idx]
                score_samp = self.get_score(torch.tensor([0]).to(c.device), c)
                if score_samp > score_mem_prevalent.min(): 
                    idx = score_mem_prevalent.argmin()
                    self.remove_from_mem(prevalenet_samp_idx[idx])
                    self.add_to_mem(xx, fea[i], prob[i])   
    
    def add_to_mem(self, x, fea, prob):
        self.mem.append(x)
        self.feats_mem = torch.cat([self.feats_mem, fea.unsqueeze(0)]) if self.feats_mem is not None else fea.unsqueeze(0)
        self.pred_prob_mem = torch.cat([self.pred_prob_mem, prob.unsqueeze(0)]) if self.pred_prob_mem is not None else prob.unsqueeze(0)
        c = prob.argmax().item()
        self.class_mem = torch.cat([self.class_mem, torch.tensor([c])]) if self.class_mem is not None else torch.tensor([c])
        self.age_mem = torch.cat([self.age_mem, torch.tensor([0])]) if self.age_mem is not None else torch.tensor([0])
        
    def remove_from_mem(self, idx):
        self.mem.pop(idx)
        self.feats_mem = torch.cat([self.feats_mem[:idx], self.feats_mem[idx+1:]])
        self.pred_prob_mem = torch.cat([self.pred_prob_mem[:idx], self.pred_prob_mem[idx+1:]])
        self.class_mem = torch.cat([self.class_mem[:idx], self.class_mem[idx+1:]])
        self.age_mem = torch.cat([self.age_mem[:idx], self.age_mem[idx+1:]])
    
    def get_pseudo_label(self, feature):
        feats_mem = self.feats_mem
        label_mem = self.pred_prob_mem.argmax(1)
        class_centroid = torch.zeros(self.num_classes, feats_mem.shape[1]).to(feature.device)
        for i in range(self.num_classes):
            feat_class = feats_mem[label_mem==i]
            # normalize
            feat_class = torch.nn.functional.normalize(feat_class, p=2, dim=1)
            class_centroid[i] = feat_class.mean(0)
        t = torch.exp(class_centroid @ feature.T)
        t = t / t.sum(0)
        return t.T
    
    def get_mem_as_tensor(self):
        if len(self.mem) == 0:
            return torch.tensor([])
        return torch.stack(self.mem)
        
class PLTA(nn.Module):
    def __init__(self, model, num_classes, lr=0.001, optim='Adam', epsilon=2/255, reverse_kl=True, conf_threshold=0.7, memory_size=20, temperature_scaling=2, lambda_pi=1):
        super(PLTA, self).__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.epsilon = epsilon
        self.reverse_kl = reverse_kl
        self.conf_threshold = conf_threshold
        self.temperature_scaling = temperature_scaling
        self.lambda_pi = lambda_pi
        self.mem = MyMem(num_classes, memory_size, conf_threshold)

        self.model = model
        params, _ = collect_params(self.model)

        if optim == 'SGD':
            self.optimizer = torch.optim.SGD(params, lr=lr)
        elif optim == 'Adam':
            self.optimizer = torch.optim.Adam(params, lr=lr)
        else:
            raise NotImplementedError
    
    def get_adv_loss(self, x, pred):
        x_adv = x.detach()
        x_adv.requires_grad_()
        pred_adv = self.model(x_adv)
        loss_adv = KL(pred, pred_adv, T=self.temperature_scaling, reverse=self.reverse_kl)
        grad = torch.autograd.grad(loss_adv.mean(), [x_adv])[0]
        x_adv = x_adv + self.epsilon*torch.sign(grad)
        x_adv = x_adv.detach()
        pred_adv = self.model(x_adv)
        return KL(pred, pred_adv, T=self.temperature_scaling, reverse=self.reverse_kl)
    
    @torch.enable_grad
    def forward(self, x):
        self.model.eval()
        featurizer, classifier = split_up_model(self.model)
        with torch.no_grad():
            output_test = torch.zeros(x.shape[0], self.num_classes).to(x.device)
            output_conf = torch.zeros(x.shape[0]).to(x.device)
            features = []
            for (i, xx) in enumerate(x):
                xx = xx.unsqueeze(0)
                fea = featurizer(xx).reshape(1, -1)
                features.append(fea)
                output_test[i] = classifier(fea)
                p = output_test[i].unsqueeze(0).softmax(1)
                output_conf[i] = p.max(1)[0]
                # update the age for all samples in the memory
                self.mem.update_mem_age()
                if output_conf[i] < self.conf_threshold and len(self.mem) > 0:
                    # replace output with low confidence by the pseudo label
                    output_test[i] = self.mem.get_pseudo_label(fea)
                else:
                    self.mem.update(xx, fea, p, output_conf[i].unsqueeze(0))
        
        self.model.train()        
        pred = self.model(x)
        ent = softmax_entropy(pred, T=self.temperature_scaling)
        score_ent = torch.exp((output_conf-1)/self.conf_threshold)
        mean_prob_ent = mean_softmax_entropy(pred, T=self.temperature_scaling, weight=score_ent)
        loss_ent = ent.mul(score_ent) - mean_prob_ent
        loss_adv = self.get_adv_loss(x, pred).mul(score_ent)
        loss = (self.lambda_pi*loss_adv+loss_ent).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return output_test

    @staticmethod
    def configure_model(model):
        # train mode, because tent optimizes the model to minimize entropy
        model.train()
        return model
    

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        for np, p in m.named_parameters():
            if np in ['weight', 'bias']:  # weight is scale, bias is shift
                params.append(p)
                names.append(f"{nm}.{np}")
    return params, names


def KL(logit1, logit2, T, reverse=False):
    if reverse:
        logit1, logit2 = logit2, logit1
    logit1 = logit1/T
    logit2 = logit2/T
    p1 = logit1.softmax(1)
    logp1 = logit1.log_softmax(1)
    logp2 = logit2.log_softmax(1) 
    return (p1*(logp1-logp2)).mean(1)

def softmax_entropy(x, T=2):
    x = x/ T
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

def mean_softmax_entropy(x, T=2, weight=None):
    if weight is None:
        weight = torch.ones(x.shape[0]).to(x.device)
    x = x / T
    mean_probe_d=torch.mean(x.softmax(1).mul(weight.unsqueeze(1)), dim=0)
    entropy=-torch.sum(mean_probe_d*torch.log(mean_probe_d))
    return entropy
