import torch
import torch.nn as nn
from copy import deepcopy
from .tent import softmax_entropy, mean_softmax_entropy
from torch.amp import autocast, GradScaler
scaler = GradScaler('cuda')
scaler_adv = GradScaler('cuda')

class TIPI(nn.Module):
    def __init__(self, model, lr=0.001, optim='SGD', epsilon=2/255, random_init_adv=False, reverse_kl=True,  tent_coeff=1, use_test_bn_with_large_batches=False):
        super(TIPI, self).__init__()

        self.lr = lr
        self.epsilon = epsilon
        self.random_init_adv = random_init_adv
        self.reverse_kl = reverse_kl
        self.tent_coeff = tent_coeff
        self.use_test_bn_with_large_batches = use_test_bn_with_large_batches
        self.large_batch_threshold = 64

        configure_multiple_BN(model,["main","adv"]) 
        self.model = model
        params, _ = collect_params(self.model)

        if optim == 'SGD':
            self.optimizer = torch.optim.SGD(params, lr=lr)
        elif optim == 'Adam':
            self.optimizer = torch.optim.Adam(params, lr=lr)
        else:
            raise NotImplementedError
        
    def get_adv_loss(self, x, pred, delta):
        use_BN_layer(self.model,'adv')
        if self.random_init_adv:
            delta = (torch.rand_like(x)*2-1) * self.epsilon
            delta.requires_grad_()
            pred_adv = self.model(x+delta)
        else:
            pred_adv = pred

        loss = KL(pred.detach(), pred_adv, reverse=self.reverse_kl).mean()
        grad = torch.autograd.grad(scaler_adv.scale(loss), [delta], retain_graph=(self.tent_coeff!=0.0) and (not self.random_init_adv))[0]
        delta = delta.detach() + self.epsilon*torch.sign(grad.detach())
        delta = torch.clip(delta,-self.epsilon,self.epsilon)
        x_adv = x + delta
        
        pred_adv = self.model(x_adv)
        return KL(pred.detach(), pred_adv, reverse=self.reverse_kl)

    @torch.enable_grad
    def forward(self, x):
        with torch.no_grad():
            if self.use_test_bn_with_large_batches and x.shape[0] > self.large_batch_threshold:
                output_test = self.model(x)
            else:
                self.model.eval()
                output_test = self.model(x)
                
        with autocast('cuda'):
            self.model.train()
            use_BN_layer(self.model,'main')
            delta = torch.zeros_like(x)
            delta.requires_grad_()
            pred = self.model(x+delta)
            
            loss_adv = self.get_adv_loss(x, pred, delta)
            loss = loss_adv.mean(0)
                        
            self.optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            use_BN_layer(self.model,'main')


        return output_test


class MultiBatchNorm2d(nn.Module):
    def __init__(self, bn, BN_layers=['main']):
        super(MultiBatchNorm2d, self).__init__()
        self.weight = bn.weight
        self.bias = bn.bias
        self.BNs = nn.ModuleDict()
        self.current_layer = 'main'
        for l in BN_layers:
            m = deepcopy(bn)
            m.weight = self.weight
            m.bias = self.bias
            try:
                self.BNs[l] = m
            except Exception:
                import pdb; pdb.set_trace()
    def forward(self,x):
        assert self.current_layer in self.BNs.keys()
        return self.BNs[self.current_layer](x)

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, MultiBatchNorm2d)\
                or isinstance(m,nn.GroupNorm)\
                or isinstance(m,nn.InstanceNorm2d)\
                or isinstance(m,nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def configure_multiple_BN(net, BN_layers=['main']):
     for name, child in net.named_children():
        if isinstance(child, nn.BatchNorm2d):
            new_bn = MultiBatchNorm2d(child, BN_layers)
            setattr(net, name, new_bn)
        else:
            configure_multiple_BN(child, BN_layers)

def use_BN_layer(net, BN_layer='main'):
    for m in net.modules():
        if isinstance(m, MultiBatchNorm2d):
            m.current_layer = BN_layer



def KL(logit1,logit2,reverse=False):
    if reverse:
        logit1, logit2 = logit2, logit1
    p1 = logit1.softmax(1)
    logp1 = logit1.log_softmax(1)
    logp2 = logit2.log_softmax(1) 
    return (p1*(logp1-logp2)).mean(1)