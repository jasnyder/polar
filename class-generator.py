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

if __name__=='__main__':
    x = 3
    fun = lambda x : x**2
    foo = MyClass(x, fun)
    for i in range(50):
        foo.apply()
        foo.grad_step()
        print(foo.x)