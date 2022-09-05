import yaml

cfg = [
    ['Conv', 3, 32, True, True, 1, 2, 0],#208
    ['Conv', 32, 64, True, True, 3, 1, 0],#206
    ['ResidualBlock', 64, 1, True],
    ['Conv', 64, 128, True, True, 3, 2, 0],#102
    ['ResidualBlock', 128, 2, True],
    ['Conv', 128, 256, True, True, 3, 2, 0],#50
    ['ResidualBlock', 256, 8, True],
    ['Conv', 256, 512, True, True, 3, 2, 0],#24
    ['Detect', 512, 4],
    ['Conv', 512, 256, True, True, 4, 2, 0],#10
    ['UpSample', 5],
    ['Connect', [1, 'x', 1]],
    ['Detect', 256, 4],
    ['Conv', 256, 128, True, True, 1, 1, 0],#10
    ['UpSample', 2],
    ['Connect', [2, 'x', 1]],
    ['Detect', 128, 4]
]

if __name__ == "__main__":
    with open('cfg.yaml', 'w') as w:
        yaml.dump(cfg, w)
