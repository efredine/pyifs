import random
import sys
import argparse
from math import cos, sin, pi, atan, atan2, sqrt, exp
from datetime import datetime

from image import Image

# CUSTOMIZE
WIDTH = 1024
HEIGHT = 1024
ITERATIONS = 10000
NUM_POINTS = 1000

NUM_TRANSFORMS = 7

def random_complex():
    return complex(random.uniform(-1, 1), random.uniform(-1, 1))
    
def generate_seed():
    now = datetime.utcnow() - datetime.utcfromtimestamp(0)
    return (now.days, now.seconds, now.microseconds)

class IFS:
    
    def __init__(self, transforms = []):
        self.transforms = transforms
        self.total_weight = sum([x for (x,y) in transforms])
    
    def add(self, transform):
        weight = random.gauss(1, 0.15) * random.gauss(1, 0.15)
        self.total_weight += weight
        self.transforms.append((weight, transform))
    
    def choose(self):
        w = random.random() * self.total_weight
        running_total = 0
        for weight, transform in self.transforms:
            running_total += weight
            if w <= running_total:
                return transform
    
    def final_transform(self, px, py):
        a = 0.5
        b = 0
        c = 0
        d = 1
        z = complex(px, py)
        z2 = (a * z + b) / (c * z + d)
        return z2.real, z2.imag
        
    def __repr__(self):
        return "%s([\n%s])" % (self.__class__.__name__, ',\n'.join(["\t(%s,%s)" % (x, repr(y)) for (x,y) in self.transforms]) )


class Transform(object):
    
    def __init__(self, params={}):
        self.params = {}
        self.initial_params = params
        self.seteither('red', params, random.random())
        self.seteither('green', params, random.random())
        self.seteither('blue', params, random.random())
    
    def seteither(self, name, params, value):
        if params.has_key(name):
            setattr(self, name, params[name])
        else:
            setattr(self, name, value)
        self.params[name] = value
    
    def transform_colour(self, r, g, b):
        r = (self.red + r) / 2
        g = (self.green + g) / 2
        b = (self.blue + b) / 2
        return r, g, b
        
    def get_new_transform(self, params={}):
        return self.__class__(dict(self.initial_params.items() + params.items()))
    
    def __repr__(self):
        return "%s(params=%s)" % (self.__class__.__name__, repr(self.params));

class LinearCenter(Transform):
    def __init__(self, params={}):
        super(LinearCenter, self).__init__(params)
        self.seteither('a', params, random.uniform(-1, 1))
        self.seteither('b', params, random.uniform(-1, 1))
        self.seteither('c', params, random.uniform(-1, 1))
        self.seteither('d', params, random.uniform(-1, 1))

    def transform(self, px, py):
        return (self.a * px + self.b * py, self.c * px + self.d * py)


class Linear(Transform):
    def __init__(self, params={}):
        super(Linear, self).__init__(params)
        self.seteither('a', params, random.uniform(-1, 1))
        self.seteither('b', params, random.uniform(-1, 1))
        self.seteither('c', params, random.uniform(-1, 1))
        self.seteither('d', params, random.uniform(-1, 1))
        self.seteither('e', params, random.uniform(-1, 1))
        self.seteither('f', params, random.uniform(-1, 1))
             
    def transform(self, px, py):
        return (self.a * px + self.b * py + self.c, self.d * px + self.e * py + self.f)


class ComplexTransform(Transform):
    def __init__(self, params={}):
        super(ComplexTransform, self).__init__(params)
    
    def transform(self, px, py):
        z = complex(px, py)
        z2 = self.complex_transform(z)
        return z2.real, z2.imag

class Moebius(ComplexTransform):
    
    def __init__(self, params={}):
        super(Moebius, self).__init__(params)
        self.seteither('a', params, random_complex())
        self.seteither('b', params, random_complex())
        self.seteither('c', params, random_complex())
        self.seteither('d', params, random_complex())
     
    def complex_transform(self, z):
        return (self.a * z + self.b) / (self.c * z + self.d)

class InverseJulia(ComplexTransform):
    
    def __init__(self, params={}):
        super(InverseJulia, self).__init__(params)
        r = sqrt(random.random()) * 0.4 + 0.8
        theta = 2 * pi * random.random()
        self.seteither('c', params, complex(r * cos(theta), r * sin(theta)))
    
    def complex_transform(self, z):
        z2 = self.c - z
        theta = atan2(z2.imag, z2.real) * 0.5
        sqrt_r = random.choice([1, -1]) * ((z2.imag * z2.imag + z2.real * z2.real) ** 0.25)
        return complex(sqrt_r * cos(theta), sqrt_r * sin(theta))

#
# COMBINE WITH LINEAR
# The following simple function transformations are best used when wrapped or sequenced with a linear transform.
#
class Identity(Transform):

    def transform(self, x, y):
        r2 = x*x + y*y
        return x, y

class Swap(Transform):
    
    def transform(self, x, y):
        return y, x
    

class Spherical(Transform):

    def transform(self, x, y):
        r2 = x*x + y*y
        if r2 == 0: r2 = 1.0
        return x/r2, y/r2

class Sinusoidal(Transform):

    def transform(self, x, y):
        return sin(x), sin(y)
        
class Swirl(Transform):

    def transform(self, x, y):
        r2 = x * x + y * y
        return x * sin(r2) - y * cos(r2), x * cos(r2) + y * sin(r2)

class HorseShoe(Transform):
 
    def transform(self, x, y):
        r = sqrt(x*x + y*y)
        return 1/r * ((x-y)*(x+y)), 1/r * 2*x*y
        
class Polar(Transform):
    
    def transform(self, x, y):
        theta = atan(x/y)
        r = sqrt(x*x + y*y)
        return theta, (r - 1)
        
class Handkerchief(Transform):
    
    def transform(self, x, y):
        theta = atan(x/y)
        r = sqrt(x*x + y*y)
        return r * sin(theta + r), r * cos(theta - r)
        
class Heart(Transform):
    
    def transform(self, x, y):
        theta = atan(x/y)
        r = sqrt(x*x + y*y)
        return r * sin(theta * r), -1 * r * cos(theta * r)
        
class Disc(Transform):

    def transform(self, x, y):
        theta = atan(x/y)
        r = sqrt(x*x + y*y)
        return theta * sin(pi * r), theta * cos(pi * r)
        
class Spiral(Transform):
    
    def transform(self, x, y):
        theta = atan(x/y)
        r = sqrt(x*x + y*y)
        return 1/r * (cos(theta) + sin(r)), 1/r *(sin(theta) - cos(r))

class Hyperbolic(Transform):
    
    def transform(self, x, y):
        theta = atan(x/y)
        r = sqrt(x*x + y*y)
        return  sin(theta)/r, random.choice([1,-1]) * r * cos(theta)
        

class Diamond(Transform):
    
    def transform(self, x, y):
        theta = atan(x/y)
        r = sqrt(x*x + y*y)
        return  2 * sin(theta) * cos(r), 2 * random.choice([1,-1]) * cos(theta) * sin(r)    

class Ex(Transform):
    
    def transform(self, x, y):
        theta = atan(x/y)
        r = sqrt(x*x + y*y)
        p0 = sin(theta+r)
        p1 = cos(theta-r)
        return  r *(p0*p0*p0 + p1*p1*p1), r*(p0*p0*p0 - p1*p1*p1)
        
class Bent(Transform):
    
    def transform(self, x, y):
      if x >= 0 and y >= 0:
           return x,y
      elif x < 0 and y >= 0:
          return 2*x, y
      elif x >= 0 and y < 0:
          return x, y/2
      else:
          return 2*x, y/2

# PARAMETRIC FUNCTIONS
# Perspective is useful as a wrapping function.
class Perspective(Transform):
    
    def __init__(self, params={}):
        super(Perspective, self).__init__(params)
        self.seteither('p1', params, 2 * pi * random.random()) #angle
        self.seteither('p2', params, random.uniform(-1, 1)) #distance
       
    def transform(self, x, y):
        coefficient = self.p2 / (self.p2 - y * sin(self.p1))
        return coefficient * x, coefficient * y * cos(self.p1)

class Rectangle(Transform):

    def __init__(self, params={}):
        super(Rectangle, self).__init__(params)
        self.seteither('p1', params, random.uniform(-1, 1)) 
        self.seteither('p2', params, random.uniform(-1, 1))

    def transform(self, x, y):
        return (2*int(x/self.p1) + 1)*self.p1 - x, (2*int(y/self.p2) + 1)*self.p2 - y
 
# Strange attractors

class Lorenz(Transform):
    def __init__(self, params={}):
        super(Rectangle, self).__init__(params)
        self.seteither('delta', params, float(10)) 
        self.seteither('rl', params, float(28))
        self.seteither('hl', params, 1e-3)
        self.seteither('hl', params, random.uniform(-1,1))

    def transform(self, x, y):
        return (2*int(x/self.p1) + 1)*self.p1 - x, (2*int(y/self.p2) + 1)*self.p2 - y
    
    
 
# Noise injection
class RandomMove(Transform):
    
    def transform(self, x, y):
        return random.uniform(-2, 2), random.uniform(-2, 2)

class TriangularMove(Transform):

    def transform(self, x, y):
        return random.triangular(-2, 2), random.triangular(-2, 2)

class ParetoMove(Transform):

    def transform(self, x, y):
        return random.choice([1,-1]) * (random.paretovariate(1)-1), random.choice([1,-1]) * (random.paretovariate(1)-1)

class ExpMove(Transform):

    def transform(self, x, y):                
        return random.choice([1,-1]) * (random.expovariate(1)), random.choice([1,-1]) * (random.expovariate(1))

class BetaMove(Transform):

    def transform(self, x, y):        
        return random.choice([2,-2]) * (random.betavariate(1.0,1.0)), random.choice([2,-2]) * (random.betavariate(1.0,1.0))

#Random walks
class Wiener(Transform):
    def __init__(self, params={}):
        super(Wiener, self).__init__(params)
        self.sigma = 2.0 ** 2.0/10000
        self.x = 0.0
        self.y = 0.0
        print self.sigma
        # self.sigma = 0.01
    
    def transform(self, x, y):
        self.x = random.gauss(0.0, self.sigma)
        self.y = random.gauss(0.0, self.sigma)
        return self.x, self.y
            
class RandomWalk(Transform):
    def __init__(self, params={}):
        super(RandomWalk, self).__init__(params)
        self.step = 1.0
        self.x = 0.0
        self.y = 0.0

    def transform(self, x, y):
        self.x = self.x + random.choice([-1.0 * self.step, 0.0, self.step])
        self.y = self.x + random.choice([-1.0 * self.step, 0.0, self.step])
        return self.x, self.y

# Sequentially applies transform functions in sequence.
# Each transform has a probability with which it should be applied - note that all functions have a chance to
# run each time this transform is run and if they all have prob = 1.0 they will always all run.
class Sequence(Transform):
    def __init__(self, params={}):
        super(Sequence, self).__init__(params)
        sequence = [(prob, t.get_new_transform()) for (prob, t) in params['sequence']]
        self.params['sequence'] = sequence
        setattr(self, 'sequence', sequence)
                
    def transform(self, px, py):
        for (prob, instance) in self.sequence:
            if prob > random.random():
                px,py = instance.transform(px,py)
        return px,py
                        
class Generator(object):

    @classmethod
    def from_open_file(cls, file_handle):
        s = file_handle.read()
        return eval(s)
        
    @classmethod
    def from_file(cls, file_name):
        f = open(file_name)
        g = Generator.from_open_file(f)
        f.close()
        return g
    
    def __init__(self, 
            ifs,
            instance="",
            seed = generate_seed(), 
            width = WIDTH, 
            height = HEIGHT, 
            iterations = ITERATIONS,
            num_points = NUM_POINTS,
            img_name = "test.png",
            before=[],
            after=[]):
        
        self.ifs = ifs 
        
        self.instance = instance
        if self.instance:
            self.instance += ":"
        else:
            self.instance = ""
        self.seed = seed
        self.width = width
        self.height = height
        self.iterations = iterations
        self.num_points = num_points
        self.img_name =  img_name  
        self.before = before
        self.after = after
        
        self.img = Image(self.width, self.height)
              
    def generate(self):
        random.seed(self.seed)
        self.report_cycle = self.num_points / 10
        for i in range(self.num_points):
            if i % self.report_cycle == 0:
                print "%s%i" % (self.instance, i/self.report_cycle*10) + '%'
            px = random.uniform(-1, 1)
            py = random.uniform(-1, 1)
            r, g, b = 0.0, 0.0, 0.0

            for j in range(self.iterations):
                t = self.ifs.choose()
                for before_transform in self.before:
                    px, py = before_transform.transform(px, py)
                px, py = t.transform(px, py)
                for after_transform in self.after:
                    px, py = after_transform.transform(px, py)
                r, g, b = t.transform_colour(r, g, b)

                fx, fy = self.ifs.final_transform(px, py)
                x = int((fx + 1) * self.width / 2)
                y = int((fy + 1) * self.height / 2)

                self.img.add_radiance(x, y, [r, g, b])  
                          
        self.img.save(self.img_name, max(1, (self.num_points * self.iterations) / (self.height * self.width)))   
             
    def __repr__(self):
        return "%s(\n%s,\nseed=%s, width=%s, height=%s, iterations=%s, num_points=%s, img_name='%s', before=%s, after=%s)" %(
            self.__class__.__name__,
            repr(self.ifs),
            self.seed,
            self.width,
            self.height,
            self.iterations,
            self.num_points,
            self.img_name,
            repr(self.before),
            repr(self.after))    
            
    def get_parameters(self):
        return {
            'seed': self.seed,
            'width': self.width,
            'height': self.height,
            'iterations': self.iterations,
            'num_points': self.num_points,
            'img_name': self.img_name,
            'before': self.before,
            'after': self.after
        }
                                             
TRANSFORM_CHOICES = [Sequence(params={'sequence':[(1.0, Linear()), (1.0, Disc()), (1.0, Perspective())]}), InverseJulia()]

def generate_ifs():
    ifs = IFS()
    for n in range(NUM_TRANSFORMS):
        transform = random.choice(TRANSFORM_CHOICES)
        ifs.add(transform.get_new_transform())
    return ifs
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", help="Input file containing an IFS generator description.  If no file is provided, an IFS generator is randomly constructed from TRANSFORM_CHOICES.", type=file)
    parser.add_argument("--desc", help="Output file where description of generated IFS is written.", type=argparse.FileType('w'), default="test.ifs")
    parser.add_argument("--image", help="Name of image file.", default="test.png")
    parser.add_argument("--instance", help="Instance identifier.")
    args = parser.parse_args()
     
    if args.gen:
        print 'Using generator file.'
        g = Generator.from_open_file(args.gen)
    else:
        print 'Creating randomly selected generator.'
        ifs = generate_ifs()
        g = Generator(ifs=ifs, img_name=args.image, instance=args.instance, before=[Moebius()], after=[Moebius()])
    g.generate()
    
    args.desc.write(repr(g))
 