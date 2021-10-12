class MyClass:
    def __init__(self, x):
        self.x = x

    def jen(self, num):
        while True:
            self.x = self.x + num
            yield self.x
    def g(self, h = lambda *args : True):
        if h(self, 0) == True:
            print('win')
        else:
            print('lose')


def f(c):
    c.x += 1
    return

def main():
    x = float(input('x = ? '))
    c = MyClass(x)
    h = lambda foo, bar : foo.x>0
    c.g(h)


if __name__ == '__main__':
    main()