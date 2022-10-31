import sys


def print_model(model, args, form, rank, index):
    print('{}  {:<5}{:>20}{:>5}{:>10}    -    {:<25} \n'.format(f'\033[1;39m', f"{index}", f"{form}", f"{rank}",
                                                                f"{model}",
                                                                f"{args}"))


class Logger:
    def __init__(self):
        super(Logger, self).__init__()
        self.data = ''

    def __call__(self, *args, **kwargs):
        sys.stdout.write(f"\r {self.data}")

    def set_desc(self, *args):
        self.data = ''.join(*(d for d in args))

    def end(self):
        sys.stdout.write('\n')

    def flush(self):
        sys.stdout.flush()
