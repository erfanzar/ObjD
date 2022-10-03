import yaml
from colorama import Fore

cfg = [
    {'name': 'Conv',
     'attributes': [3, 32, 3, 1, True, True]},
    {'name': 'Conv',
     'attributes': [32, 64, 3, 2, True, True]},
    {'name': 'ResidualBlock',
     'attributes': [64, 2, True]},
    {'name': 'C3',
     'attributes': [64, 128, True, 2, 0.8]},
    {'name': 'Conv',
     'attributes': [128, 256, 3, 2, True, True]},
    {'name': 'Conv',
     'attributes': [256, 256, 3, 2, True, True]},
    {'name': 'MP',
     'attributes': []},  # 52
    {'name': 'ResidualBlock',
     'attributes': [256, 6, True]},
    {'name': 'RepConv',
     'attributes': [256, 1, 2]},
    {'name': 'Conv',
     'attributes': [256, 384, 3, 2, True, True]},
    {'name': 'MP',
     'attributes': []},  # 26
    {'name': 'ResidualBlock',
     'attributes': [384, 4, True]},
    {'name': 'Conv',
     'attributes': [384, 384, 3, 2, True, True]},
    {'name': 'Conv',
     'attributes': [384, 728, 3, 1, True, True]},
    {'name': 'Detect',
     'attributes': [728, 4]},  # Detect 13
    {'name': 'UpSample',
     'attributes': [2]},  # up to 26
    {'name': 'Conv',
     'attributes': [1112, 1024, 3, 1, True, True]},
    {'name': 'Detect',
     'attributes': [1024, 4]},  # Detect 26
    {'name': 'UpSample',
     'attributes': [2]},  # up to 52
    {'name': 'Conv',
     'attributes': [1280, 1280, 3, 1, True, True]},
    {'name': 'Detect',
     'attributes': [1280, 4]},  # Detect 52
]

if __name__ == "__main__":
    print(f"{Fore.BLUE}")
    with open('../cfg/cfg.yaml', 'w') as w:
        yaml.dump(cfg, w)
