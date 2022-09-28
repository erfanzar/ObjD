import yaml
from colorama import Fore

cfg = [
    {'name': 'Conv',
     'attributes': [3, 32, 3, 1, True, True]},
    {'name': 'Conv',
     'attributes': [32, 64, 3, 2, True, True]},
    {'name': 'ResidualBlock',
     'attributes': [64, 4, True]},
    {'name': 'C3',
     'attributes': [64, 128, True, 2, 0.8]},
    {'name': 'Conv',
     'attributes': [128, 256, 3, 2, True, True]},
    {'name': 'ResidualBlock',
     'attributes': [256, 2, True]},
    {'name': 'RepConv',
     'attributes': [256, 1, 2]},
    {'name': 'Conv',
     'attributes': [256, 384, 3, 2, True, True]},
    {'name': 'Detect',
     'attributes': [384, 4]},  # Detect
    {'name': 'RepConv',
     'attributes': [384, 1, 2]},
    {'name': 'Conv',
     'attributes': [384, 512, 3, 2, True, True]},
    {'name': 'Detect',
     'attributes': [512, 4]},  # Detect
    {'name': 'RepConv',
     'attributes': [512, 1, 2]},
    {'name': 'Conv',
     'attributes': [512, 768, 3, 2, True, True]},
    {'name': 'Detect',
     'attributes': [768, 4]},  # Detect
]

if __name__ == "__main__":
    print(f"{Fore.BLUE}")
    with open('../cfg/cfg.yaml', 'w') as w:
        yaml.dump(cfg, w)
