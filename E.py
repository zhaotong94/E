import torch
class E(torch.optim.Optimizer):
    def __init__(self, base_optimizer, 
                 alpha=-5,
                 beta=0.99,
                 noise_std=0.0, 
                 grad_clip=None):
        self.alpha=alpha
        self.beta=beta
        self.opt = base_optimizer
        self.noise_std = noise_std
        self.grad_clip = grad_clip
        super().__init__(self.opt.param_groups, self.opt.defaults)
        # super().__init__([], {})
        # self._param_groups = base_optimizer.param_groups
        # self._defaults = base_optimizer.defaults
        # super().__init__([], {})
    def __setstate__(self, state):
        super().__setstate__(state)
    # @property
    # def param_groups(self):
    #     return self.opt.param_groups

    # @property
    # def defaults(self):
    #     return self.opt.defaults
    # @property
    # def state(self):
    #     return self.opt.state

    def state_dict(self):
        return self.opt.state_dict()

    def load_state_dict(self, state_dict):
        self.opt.load_state_dict(state_dict)



    def zero_grad(self):
        self.opt.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        # 前置处理
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # 加噪声
                if self.noise_std > 0:
                    grad.add_(torch.randn_like(grad) * self.noise_std)

                # 梯度裁剪
                if self.grad_clip is not None:
                    grad.clamp_(-self.grad_clip, self.grad_clip)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Store previous gradient
                    state['prev_grad'] = torch.zeros_like(p.data)
                    if sum(p.numel() for group in self.param_groups for p in group['params'])<3:
                        state['prev_grad'] = grad.clone()
                    state['exp_grad_diff'] = torch.zeros_like(p.data)
                # beta = group['beta']
                # alp = group['alp']
                if 'prev_grad' in state:
                    grad_diff = grad - state['prev_grad']
                else:
                    grad_diff = torch.zeros_like(grad)
                state['prev_grad'] = grad.clone()
                state['exp_grad_diff'].mul_(self.beta).add_(grad_diff, alpha=1 - self.beta)
                modified_grad = grad.add(state['exp_grad_diff'], alpha=self.alpha)
                p.grad = modified_grad  # 替换回去

        # 调用原优化器更新
        return self.opt.step(closure)

    # def state_dict(self):
    #     return self.opt.state_dict()

    # def load_state_dict(self, state_dict):
    #     self.opt.load_state_dict(state_dict)

    # @property
    # def param_groups(self):
    #     return self.opt.param_groups
    # @property
    # def defaults(self):
    #     return self.opt.defaults

    # def __getattr__(self, name):
    #     if "opt" in self.__dict__:
    #         return getattr(self.opt, name)

    #     raise AttributeError(f"'OptimizerAdaptor' has no attribute '{name}'")

