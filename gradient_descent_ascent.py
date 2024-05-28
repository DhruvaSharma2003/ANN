x0, y0 = 2, -2

import matplotlib.pyplot as plt

def func(x, y): 
    f = x * y + 4 * y - 3 * x * x - y * y 
    return f

def func1(x, y): 
    fxd = y - 6 * x 
    return fxd

def func2(x, y): 
    fyd = x + 4 - 2 * y 
    return fyd

n = 0.01 

def nextpt(x, f): 
    np = x + n * f 
    return np

def nextpt1(y, f): 
    np = y + n * f 
    return np

x1, y1 = nextpt(x0, func1(x0, y0)), nextpt1(y0, func2(x0, y0)) 
print(x1) 
print(y1)

#Ascent

xx = [] 
yy = [] 
xx.append(2) 
yy.append(-2)

for i in range(10): 
    xx.append(nextpt(xx[i], func1(xx[i], yy[i]))) 
    yy.append(nextpt1(yy[i], func2(xx[i], yy[i]))) 
    print(xx[i+1], yy[i+1])

ff = [] 
for i in range(11): 
    ff.append(func(xx[i], yy[i]))

plt.plot(range(11), ff) 
plt.xlabel('Epoch') 
plt.ylabel('values') 
plt.title("") 
plt.grid(True) 
plt.show()

# Descent 

xx1 = [] 
yy1 = [] 
xx1.append(2) 
yy1.append(-2)

for i in range(10): 
    xx1.append(nextpt(xx1[i], -func1(xx1[i], yy1[i]))) 
    yy1.append(nextpt1(yy1[i], -func2(xx1[i], yy1[i]))) 
    print(xx1[i+1], yy1[i+1])

ff1 = [] 
for i in range(11): 
    ff1.append(func(xx1[i], yy1[i]))

plt.plot(range(11), ff1) 
plt.xlabel('Epoch') 
plt.ylabel('values') 
plt.title("") 
plt.grid(True) 
plt.show()
