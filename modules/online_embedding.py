import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional

from torch import Tensor
import torch.nn.init as init
from torch.optim import _functional as optim_F


if torch.cuda.is_available():
    torch.classes.load_library("build/libmerlin_kv.so")


class OnlineParameter(nn.Parameter):

    def __new__(cls, data=None, requires_grad=True):
        _self = super(OnlineParameter, cls).__new__(cls, data, requires_grad)
        _self.embedding_dim = data.shape[-1]
        try:
            _self._hashtable = torch.classes.merlin_kv.HashTable()
            _self._hashtable.init(8192, 8192, 1024 * 1024 * 1024 * 16, 0.75)
        except:
            # fake hashtable for cpu debug. To be removed before regular release.
            _self._hashtable = torch.empty([1], dtype=torch.float32, device='cpu')
        return _self

    @property
    def table(self):
        return self._hashtable

    def forward(self, ids):
        self.data = torch.zeros([ids.view(-1, 1).shape[0], self.embedding_dim], dtype=torch.float32, device=self.device)
        init.normal_(self.data)
        self.ids = ids
        # print("forward ids", self.ids)
        # print("forward data", self.data)
        try:
            found = torch.zeros_like(ids, dtype=torch.bool, device=self.device)
            self.table.find(self.ids.view(-1, 1).shape[0], ids, self.data, found)
            self.data.reshape([-1, self.embedding_dim])
        except:
            pass
        return self

    def backward(self):
        # print("backward ids", self.ids)
        # print("backward data", self.data)
        try:
            self.table.insert_or_assign(self.ids.view(-1, 1).shape[0], self.ids, self.data)
        except:
            pass

    def size(self):
        return self.table.size()


class OnlineSGD(optim.SGD):
    """cutomizing the step() for creating new DynamicTensor on each step"""

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            maximize = group['maximize']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            optim_F.sgd(params_with_grad,
                        d_p_list,
                        momentum_buffer_list,
                        weight_decay=weight_decay,
                        momentum=momentum,
                        lr=lr,
                        dampening=dampening,
                        nesterov=nesterov,
                        maximize=maximize,)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

            for p in params_with_grad:
                if isinstance(p, OnlineParameter):
                    p.backward()

        return loss


def zeros_like(input: Tensor, *, memory_format: Optional[torch.memory_format] = None):
    if isinstance(input, OnlineParameter):
        return OnlineParameter(torch.empty((1, input.embedding_dim), device=input.device), requires_grad=False)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.zeros_like(input=input, memory_format=memory_format, device=device)


class OnlineAdam(optim.Adam):
    """cutomizing the step() for creating new DynamicTensor on each step"""

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    for name in state:
                        if isinstance(state[name], OnlineParameter):
                            state[name].forward(p.ids)

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            optim_F.adam(params_with_grad,
                         grads,
                         exp_avgs,
                         exp_avg_sqs,
                         max_exp_avg_sqs,
                         state_steps,
                         amsgrad=group['amsgrad'],
                         beta1=beta1,
                         beta2=beta2,
                         lr=group['lr'],
                         weight_decay=group['weight_decay'],
                         eps=group['eps'],
                         maximize=group['maximize'])
            for p in params_with_grad:
                if isinstance(p, OnlineParameter):
                    p.backward()
                    state = self.state[p]
                    for name in state:
                        if isinstance(state[name], OnlineParameter):
                            state[name].backward()

        return loss


class OnlineEmbedding(nn.Module):
    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    sparse: bool

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None,
                 device=None, dtype=None) -> None:
        """num_embeddings is no-used and remained for compatible with static embedding in pytorch."""

        factory_kwargs = {'device': device, 'dtype': dtype}
        super(OnlineEmbedding, self).__init__()

        self.num_embeddings = 2**64
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq

        self.weight = OnlineParameter(torch.empty((1, embedding_dim), **factory_kwargs))
        self.reset_parameters()

        self.sparse = sparse
        self.factory_kwargs = factory_kwargs

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        continue_input = torch.arange(0, input.view(-1, 1).size(0),
                                      device=self.factory_kwargs["device"], dtype=torch.long, requires_grad=False)
        self.weight = self.weight.forward(input)
        return F.embedding(
            continue_input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)

    @classmethod
    def from_pretrained(cls, embeddings, freeze=True, padding_idx=None,
                        max_norm=None, norm_type=2., scale_grad_by_freq=False,
                        sparse=False):
        """W.I.P"""
        pass
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = cls(
            embedding_dim=cols,
            _weight=embeddings,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse)
        embedding.weight.requires_grad = not freeze
        return embedding
