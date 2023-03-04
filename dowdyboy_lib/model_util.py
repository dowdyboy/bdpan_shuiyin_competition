from torchinfo import summary
from ptflops.flops_counter import get_model_complexity_info


def frozen_module(module):
    for key, value in module.named_parameters():  # named_parameters()包含网络模块名称 key为模型模块名称 value为模型模块值，可以通过判断模块名称进行对应模块冻结
        value.requires_grad = False


def unfrozen_module(module):
    for key, value in module.named_parameters():
        value.requires_grad = True


def module_summary(model, in_size):
    in_size = (1, ) + in_size
    summary(model, input_size=in_size, device='cpu')


def module_cost(model, in_size):
    flops, params = get_model_complexity_info(model, in_size, as_strings=False, print_per_layer_stat=True)
    print('total params : ' + str(params / 1e6) + 'M')
    print('Flops: ' + str(flops / 1e9) + 'G')

