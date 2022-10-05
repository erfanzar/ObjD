import torch.nn as nn
import torch
from module.commons import *

Any = [list, dict, int, float, str]


def name_to_layer(name: str, attr: Any = None, in_case_prefix_use=None, prefix: Any = None,
                  form: [list, int] = -1,
                  print_debug: bool = True):
    if in_case_prefix_use is None:
        in_case_prefix_use = []
    attr = [attr] if isinstance(attr, int) else attr
    t = len(attr) if isinstance(attr, list) else (attr if attr is not None else 0)
    tu = ''.join((f'{v},' if i != t else f'{v}') for i, v in enumerate(attr)) if t != 0 else ''
    pr = '' if prefix is None else (f'{prefix},' if name in in_case_prefix_use else '') if t != 0 else ''
    fr = form if t != 0 else ''
    model_str = f'{name}({pr}{tu}{fr})'
    if print_debug:
        print(f"ADDING : {model_str} ")
    model = eval(model_str)
    return model


def module_creator(backbone, head, print_status, ic_backbone,
                   ic_head):
    backbone_m = nn.ModuleList()
    head_m = nn.ModuleList()
    in_case_prefix_use = ['Conv']
    if print_status:
        print('BackBone Module **')
    for b in backbone:
        form = b[0]
        name = b[2]
        attr = attr_exist_check_(b, 3)
        ic_backbone = ic_backbone * len(form) if name == 'Concat' else ic_backbone
        backbone_m.append(
            name_to_layer(name=name, attr=attr, prefix=ic_backbone, in_case_prefix_use=in_case_prefix_use, form=form,
                          print_debug=print_status))
        if name in in_case_prefix_use:
            ic_backbone = attr[0]

    if print_status:
        print('Head Module **')
    for h in head:
        form = h[0]
        name = h[2]
        attr = attr_exist_check_(h, 3)
        ic_head = ic_head * len(form) if name == 'Concat' else ic_head
        head_m.append(
            name_to_layer(name=name, attr=attr, prefix=ic_head, in_case_prefix_use=in_case_prefix_use, form=form,
                          print_debug=print_status))
        if name in in_case_prefix_use:
            ic_head = attr[0]

    return backbone_m, head_m


def attr_exist_check_(attr, index):
    try:
        s = attr[index]
    except IndexError:
        s = []
    return s
