import torch.nn as nn
import torch
from .logger import print_model
from module.commons import *

Any = [list, dict, int, float, str]


def name_to_layer(name: str, attr: Any = None, in_case_prefix_use=None, prefix: Any = None,
                  form: [list, int] = -1,
                  print_debug: bool = True, nc: int = 4, anchors: list = None):
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


def module_creator(backbone, head, print_status, ic_backbone, nc, anchors):
    model = nn.ModuleList()

    save = []
    sv_bb = []
    in_case_prefix_use = ['Conv']
    sva = 0
    if print_status:
        print('BackBone Module **')
    for i, b in enumerate(backbone):
        sva = i
        form = b[0]
        rank = b[1]
        name = b[2]
        attr = attr_exist_check_(b, 3)
        ic_backbone = ic_backbone * len(form) if name == 'Concat' else ic_backbone
        model.append(
            name_to_layer(name=name, attr=attr, prefix=ic_backbone, in_case_prefix_use=in_case_prefix_use, form=form,
                          print_debug=print_status, nc=nc, anchors=anchors))
        print_model(name, attr, form=form, rank=rank, index=sva)
        if name in in_case_prefix_use:
            ic_backbone = attr[0]
        save.extend(x % i for x in ([form] if isinstance(form, int) else form) if x != -1)

    ic_head = ic_backbone
    sva += 1
    if print_status:
        print('Head Module **')
    for i, h in enumerate(head):
        form = h[0]
        rank = h[1]
        name = h[2]
        attr = attr_exist_check_(h, 3)
        ic_head = ic_head * len(form) if name == 'Concat' else ic_head
        model.append(
            name_to_layer(name=name, attr=attr, prefix=ic_head, in_case_prefix_use=in_case_prefix_use, form=form,
                          print_debug=print_status, nc=nc, anchors=anchors))
        print_model(name, attr, form=form, rank=rank, index=i + sva)
        if name in in_case_prefix_use:
            ic_head = attr[0]
        save.extend(x % (i + sva) for x in ([form] if isinstance(form, int) else form) if x != -1)

    return model, save


def attr_exist_check_(attr, index):
    try:
        s = attr[index]
    except IndexError:
        s = []
    return s


def iou(box1, box2):
    xma = max(box1[0], box2[0])
    yma = max(box1[1], box2[1])
    xmi = min(box1[2], box2[2])
    ymi = min(box1[3], box2[3])

    i_area = abs(max(xma - xmi, 0) * max(yma - ymi, 0))

    box1_area = abs((box1[2] - box1[0]) * (box1[3] - box1[1]))
    box2_area = abs((box2[2] - box2[0]) * (box2[3] - box2[1]))
    result = i_area / float(box2_area + box1_area - i_area)
    return result
