import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional

from torch import Tensor
import torch.nn.init as init
from torch.optim import _functional as optim_F


torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

############################################

class OnlineParameter(nn.Parameter):

  def __new__(cls, data=None, requires_grad=True):
    _self = super(OnlineParameter, cls).__new__(cls, data, requires_grad)

    try:
        torch.classes.load_library("build/libmerlin_kv.so")
        _self._hashtable = torch.classes.merlin_kv.HashTable()
        _self._hashtable.init(8192, 8192, 1024 * 1024 * 1024 * 16, 0.75)
    except:
        _self._hashtable = torch.empty([1], dtype=torch.float32, device=device)
    return _self

  @property
  def table(self):
    return self._hashtable

  def forward(self, ids):
    self.data = torch.zeros([ids.view(-1, 1).shape[0], EMBEDDING_DIM], dtype=torch.float32, device=device)
    init.normal_(self.data)
    self.ids = ids
    print("forward ids", self.ids)
    print("forward data", self.data)
    try:
        found = torch.zeros_like(ids, dtype=torch.bool, device=device)
        self.table.find(self.ids.view(-1, 1).shape[0], ids, self.data, found)
    except:
        pass
    return self

  def backward(self):
    print("backward ids", self.ids)
    print("backward data", self.data)
    try:
        self.table.insert_or_assign(self.ids.view(-1, 1).shape[0], self.ids, self.data)
    except:
        pass

  def size(self):
    try:
        return self.table.size()
    except:
        pass


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


@torch.no_grad()
def F_embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx: Optional[int] = None,
    max_norm: Optional[float] = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> Tensor:
    if max_norm is not None:
        input = input.contiguous()
    return weight.forward(input)


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

    def __init__(self, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(OnlineEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq

        self.weight = OnlineParameter()
        self.reset_parameters()

        self.sparse = sparse

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, input: Tensor) -> Tensor:
        return F_embedding(
            input, self.weight, self.padding_idx, self.max_norm,
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


###############################################

CONTEXT_SIZE = 2
EMBEDDING_DIM = 2
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.
# Each tuple is ([ word_i-CONTEXT_SIZE, ..., word_i-1 ], target word)
ngrams = [
    (
        [test_sentence[i - j - 1] for j in range(CONTEXT_SIZE)],
        test_sentence[i]
    )
    for i in range(CONTEXT_SIZE, len(test_sentence))
]
# Print the first 3, just so you can see what they look like.
print(ngrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, embedding_dim, context_size):
        vocab_size = len(vocab)
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = OnlineEmbedding(embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128, device=device)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = self.linear1(embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = OnlineSGD(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for context, target in ngrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long, device=device)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long, device=device))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)
    print(model.embeddings.weight.size())
print(losses)  # The loss decreased every iteration over the training data!

# # To get the embedding of a particular word, e.g. "beauty"
# print(model.embeddings.weight[word_to_ix["beauty"]])
