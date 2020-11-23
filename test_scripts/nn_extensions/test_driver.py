import torch
from PIL import Image
import torchvision.transforms.functional as TF
import torch.nn.functional as F

def softplus_sigmoid(input):
    return F.softplus(input) * torch.sigmoid(input)

def binary_cross_entropy_with_logits_and_regularization(input, target, img, weight=None,
        size_average=None, reduce=None, reduction='mean', pos_weight=None):
    """
    Mostly a straight copy paste of binary_cross_entropy_with_logits. Adds the term
    ```max((i-w/2)^2, (j-h/2)^2)``` to the loss function. Goes with the BCE_logits_regularized class
    """
    if not torch.jit.is_scripting():
        tens_ops = (input, target)
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(
                    binary_cross_entropy_with_logits, tens_ops, input, target, weight=weight,
                    size_average=size_average, reduce=reduce, reduction=reduction,
                    pos_weight=pos_weight
                    )
        # handle deprecated features
    if size_average is not None or reduce is not None:
        reduction_enum = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction_enum = _Reduction.get_enum(reduction)

    # probably also yell if img size != target/input size
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))



class BCE_logits_regularized(BCEWithLogitsLoss):
    """
    This is an extension of the BCEWithLogitsLoss class.
    We add the term max((i-width/2)^2, (j-height/2)^2) to each
    pixel
    """
    def __init__(self, img, weight=None, size_average=None, reduce=None, reduction: str = 'mean'
            pos_weight: Optional[Tensor] = None) -> None:
        super(BCEWithLogitsLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.register_buffer('img', img)

    def forward(self, input: Tensor, target: Tensor, img: Tensor) -> Tensor:




image = Image.open("img.jpg")
z = TF.to_tensor(image)
z.unsqueeze_(0)

dtype = torch.float
device = torch.device("cuda:0")
cuda0 = z.to(device, dtype=dtype)
cuda0.requires_grad=True

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
#N, D_in, H, D_out = 64, 1000, 100, 10
N, D_in, H, D_out = 425, 640, 100, 10

# somehow substitute with an input image - make sure above dimensions match
#x = torch.randn(N, D_in, device=device, dtype=dtype)
x = cuda0[0][0]
print(x.shape)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# random initial weights as Tensors
# use requires_grad=True to use autodiff
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
# "train"
for t in range(500):
    # forward pass
    # need to figure out how to replace with custom activation function
    h = x.mm(w1)
    h_sps = softplus_sigmoid(h)
    y_pred = h_sps.mm(w2)
    #y_pred = x.mm(w1).softplus_sigmoid(min=0).mm(w2)

    # compute loss
    # need to figure out how to replace with (custom) loss function
    #loss = (y_pred - y).pow(2).sum()
    loss = F.binary_cross_entropy_with_logits(y_pred, y, reduction="mean")
    if t % 100 == 0:
        print(t, loss.item())
        print(y_pred)
        print(loss.shape)

    # use autograd to compute backward pass
    loss.backward()

    # manually update weights with no_grad so we don't track this stuff, as
    # autograd doesn't need it
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # zero out the gradients after updating wieghts
        w1.grad.zero_()
        w2.grad.zero_()

