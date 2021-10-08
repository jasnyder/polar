class MyClass:
    def __init__(self, x):
        self.x = x

    def jen(self, num):
        i = 0
        while i < num:
            i+=1
            self.x = self.x * 2
            yield i

def main():
    x = float(input('x = ? '))
    num = int(input('num = ? '))
    c = MyClass(x)
    for j in c.jen(num):
        print(j)
    print(f'now x = {c.x}')


if __name__ == '__main__':
    main()