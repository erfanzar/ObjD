import yaml

cfg = [
    ['Conv', 3, 32, True, True, 3, 1],
    ['Conv', 32, 64, True, True, 3, 2],
    ['ResidualBlock', 64, 1, True],
    ['Conv', 64, 128, True, True, 3, 2],
    ['ResidualBlock', 128, 1, True],
    ['Conv', 128, 128, True, True, 3, 2],
    ['ResidualBlock', 128, 2, True],
    ['Conv', 128, 256, True, True, 3, 2],
    ['ResidualBlock', 256, 2, True],
    ['Conv', 256, 128, True, True, 1, 2],
    ['UpSample', 2],
    ['Detect', 384, 4],
    ['Conv', 384, 256, True, True, 3, 2],
    ['UpSample', 4],
    ['Detect', 384, 4],
    ['Conv', 384, 256, True, True, 3, 2],
    ['UpSample', 4],
    ['Detect', 384, 4]
]

if __name__ == "__main__":
    with open('cfg.yaml', 'w') as w:
        yaml.dump(cfg, w)
    print('{:>20}{:>20}{:>20}{:>20}{:>20}{:>20}{:>20}'.format('Type |', 'in_channels |', 'out_channels |', 'use act |',
                                                              'use bn |', 'kernel |', 'stride |'))
    print('-' * 20 * 7)
    for arg in cfg:
        txt = ''
        ea = 0
        for i, args in enumerate(arg):
            txt += "{:>20}".format(args)
            ea = i
        print(txt)
    print('-' * 20 * 7)
