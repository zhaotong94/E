import torch

class ALTO(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.1, 0.9, 0.99), alp=-5, eps=1e-8,
                 weight_decay=1e-4):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 2: {}".format(betas[2]))
        defaults = dict(lr=lr, betas=betas, eps=eps, alp=alp,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        torch.nn.utils.clip_grad_norm_(
            parameters=[
                p for group in self.param_groups for p in group['params']],
            max_norm=1.0,
            norm_type=2
        )

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'We does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # Store previous gradient
                    state['prev_grad'] = torch.zeros_like(p.data)

                    state['exp_grad_diff'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2, beta3 = group['betas']
                alp = group['alp']

                if 'prev_grad' in state:
                    grad_diff = grad - state['prev_grad']
                else:
                    grad_diff = torch.zeros_like(grad)

                # a_t: state['exp_grad_diff']
                state['exp_grad_diff'].mul_(beta1).add_(grad_diff, alpha=1 - beta1)

                # bias_correction1 = 1 - beta1 ** state['step']
                # if 'bias_corr_a' not in state:
                #     state['bias_corr_a'] = torch.zeros_like(p.data)
                # state['bias_corr_a'] = state['exp_grad_diff'] / bias_correction1


                modified_grad = grad.add(state['exp_grad_diff'], alpha=alp)

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta2).add_(modified_grad, alpha=1 - beta2)
                # v_t
                exp_avg_sq.mul_(beta3)
                exp_avg_sq.addcmul_(modified_grad, modified_grad, value=1 - beta3)
                
                bias_correction2 = 1 - beta2 ** state['step']
                bias_correction3 = 1 - beta3 ** state['step']
                
                if 'bias_corr_m' not in state:
                    state['bias_corr_m'] = torch.zeros_like(p.data)
                if 'bias_corr_v' not in state:
                    state['bias_corr_v'] = torch.zeros_like(p.data)

                
                state['bias_corr_m'] = exp_avg / bias_correction2
                state['bias_corr_v'] = exp_avg_sq / bias_correction3
                

                scaled_lr = group['lr']
                update = state['bias_corr_m'] / state['bias_corr_v'].sqrt().add(group['eps'])
                
                if group['weight_decay'] != 0:
                    update.add_(p.data, alpha=group['weight_decay'])
                    w_norm = torch.norm(p)
                    r_norm = torch.norm(update)
                    trust_ratio = torch.where(
                        w_norm > 0 and r_norm > 0,
                        w_norm / r_norm,
                        torch.ones_like(w_norm)
                    )
                    scaled_lr *= trust_ratio.item()
                p.data.add_(update, alpha=-scaled_lr)
                state['prev_grad'] = grad.clone()
        return loss


def create_ALTO_optimizer(model, lr, betas=(0.99, 0.9, 0.99),alpha=-5, eps=1e-6,
                          weight_decay=1e-4, exclude_layers=['bn', 'ln', 'bias']):
    # can only exclude BatchNorm, LayerNorm, bias layers
    # ['bn', 'ln'] will exclude BatchNorm, LayerNorm layers
    # ['bn', 'ln', 'bias'] will exclude BatchNorm, LayerNorm, bias layers
    # [] will not exclude any layers
    if 'bias' in exclude_layers:
        params = [
            dict(params=get_common_parameters(
                model, exclude_func=get_norm_bias_parameters)),
            dict(params=get_norm_bias_parameters(model), weight_decay=0)
        ]
    elif len(exclude_layers) > 0:
        params = [
            dict(params=get_common_parameters(
                model, exclude_func=get_norm_parameters)),
            dict(params=get_norm_parameters(model), weight_decay=0)
        ]
    else:
        params = model.parameters()
    optimizer = ALTO(params, lr, betas=betas, alp=alpha, eps=eps,
                     weight_decay=weight_decay)
    return optimizer


BN_CLS = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)


def get_parameters_from_cls(module, cls_):
    def get_members_fn(m):
        if isinstance(m, cls_):
            return m._parameters.items()
        else:
            return dict()
    named_parameters = module._named_members(get_members_fn=get_members_fn)
    for name, param in named_parameters:
        yield param


def get_bn_parameters(module):
    return get_parameters_from_cls(module, BN_CLS)


def get_ln_parameters(module):
    return get_parameters_from_cls(module, torch.nn.LayerNorm)


def get_norm_parameters(module):
    return get_parameters_from_cls(module, (torch.nn.LayerNorm, *BN_CLS))


def get_bias_parameters(module, exclude_func=None):
    excluded_parameters = set()
    if exclude_func is not None:
        for param in exclude_func(module):
            excluded_parameters.add(param)
    for name, param in module.named_parameters():
        if param not in excluded_parameters and 'bias' in name:
            yield param


def get_norm_bias_parameters(module):
    for param in get_norm_parameters(module):
        yield param
    for param in get_bias_parameters(module, exclude_func=get_norm_parameters):
        yield param


def get_common_parameters(module, exclude_func=None):
    excluded_parameters = set()
    if exclude_func is not None:
        for param in exclude_func(module):
            excluded_parameters.add(param)
    for name, param in module.named_parameters():
        if param not in excluded_parameters:
            yield param