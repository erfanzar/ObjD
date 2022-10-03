Model Architectire :



<img src="https://raw.githubusercontent.com/erfanzar/ObjD/main/Read/modelArchitecture.png"  width="500" height="600">





all the cnn modules:

------------------------------------------------------------

CONV [c1, c2, act, bn, k, s]
NECK [c1, c2, e=0.5, shortcut=False]
C3 [c1, c2, e=0.5, n=1, shortcut=True]
C4P [c, e=0.5, n=1, ct=2]
ResidualBlock [c1, n: int = 4, use_residual: bool = False]
Detect [c1, nc]
CV1 [c1, c2, e=0.5, n=1, shortcut=False]
UC1 [c1, c2, e=0.5, dim=-1]
MP [] # Creating A New Route or add to new one

------------------------------------------------------------

- c1 : income channels of an image eg : shape (1,3,640,640)
- c2 : output of image channels eg : shape(1,3,640,640) to shape(1,64,640,640)
- e : num to * or divide by c2 or c1 customized by cnn neurons by def set to c2
- n : number of times to repeat sequential neurons
- mp : add connection route
- k : kernel_size
- act : activation function on module
- bn : batch normalization on module
- s : stride
