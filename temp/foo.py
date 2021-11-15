import sys


def bar():
    print('bar!')
def baz():
    print("BAZ!!")

if __name__ == '__main__':
    globals()[sys.argv[1]]()