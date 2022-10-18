import torch.nn as nn
import torch
from .logger import print_model
from module.commons import *
import sys
import math

Any = [list, dict, int, float, str]


def printf(*args, end: bool = False):
    sys.stdout.write(''.join(f'{a}' for a in args))
    if end:
        print('\n')


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
    print(box1)
    print(box2)
    xma = max(box1[..., 0], box2[..., 0])
    yma = max(box1[..., 1], box2[..., 1])
    xmi = min(box1[..., 2], box2[..., 2])
    ymi = min(box1[..., 3], box2[..., 3])

    i_area = abs(max(xma - xmi, 0) * max(yma - ymi, 0))

    box1_area = abs((box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1]))
    box2_area = abs((box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1]))
    result = i_area / float(box2_area + box1_area - i_area)
    return result


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    box1, box2 = box1.cpu(), box2.cpu()
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU
