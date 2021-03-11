from ...core.utils.utils import list_modules
from ...framework import step


def list_steps():
    steps = list_modules(step)
    return steps


if __name__ == '__main__':
    steps = list_steps()
    print(steps)