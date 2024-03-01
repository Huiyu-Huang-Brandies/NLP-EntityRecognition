c = 0
def m(x, y, c):
    if c<5:
        t = x
        x = x-0.1*(4*x-2*x*y)
        y = y-0.1*(-t*t+2*y)
        c += 1
        print(x, y, c)
        return m(x,y,c)

m(1,1,0)