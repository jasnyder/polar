import itertools
import torch

class MyClass():
    def __init__(self, x, fun, step = 0.1):
        self.x = torch.tensor(x, dtype = torch.float, requires_grad=True)
        self.f = fun
        self.step = step
        self.y = None
    def apply(self):
        self.y = self.f(self.x)
    def grad_step(self):
        self.y.backward()
        with torch.no_grad():
            self.x += -self.step*self.x.grad
        self.x.grad.zero_()

class A:
    def __init__(self, x, a = 3):
        self.a = a
        self.x = x

class B(A):
    def __init__(self, *args, b = 2, **kwargs):
        self.b = b
        super().__init__(*args, **kwargs)


if __name__=='__main__':
    bb = B('x', a = 6, b = 8)
    print(bb.__dict__)