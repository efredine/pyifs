import random
import sys
import argparse
from math import cos, sin, pi, atan2, sqrt
from datetime import datetime

from image import Image

# CUSTOMIZE
WIDTH = 512
HEIGHT = 512
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
    
    def __repr__(self):
        return "%s(params=%s)" % (self.__class__.__name__, repr(self.params));

class Linear(Transform):
    def __init__(self, params={}):
        super(Linear, self).__init__(params)
        self.seteither('a', params, random.uniform(-1, 1))
        self.seteither('b', params, random.uniform(-1, 1))
        self.seteither('c', params, random.uniform(-1, 1))
        self.seteither('d', params, random.uniform(-1, 1))
             
    def transform(self, px, py):
        return (self.a * px + self.b * py, self.c * px + self.d * py)


class ComplexTransform(Transform):
    def __init__(self, params={}):
        super(ComplexTransform, self).__init__(params)
    
    def transform(self, px, py):
        z = complex(px, py)
        z2 = self.f(z)
        return z2.real, z2.imag


class Moebius(ComplexTransform):
    
    def __init__(self, params={}):
        super(Moebius, self).__init__(params)
        self.seteither('a', params, random_complex())
        self.seteither('b', params, random_complex())
        self.seteither('c', params, random_complex())
        self.seteither('d', params, random_complex())
     
    def f(self, z):
        return (self.a * z + self.b) / (self.c * z + self.d)

class InverseJulia(ComplexTransform):
    
    def __init__(self, params={}):
        super(InverseJulia, self).__init__(params)
        r = sqrt(random.random()) * 0.4 + 0.8
        theta = 2 * pi * random.random()
        self.seteither('c', params, complex(r * cos(theta), r * sin(theta)))
    
    def f(self, z):
        z2 = self.c - z
        theta = atan2(z2.imag, z2.real) * 0.5
        sqrt_r = random.choice([1, -1]) * ((z2.imag * z2.imag + z2.real * z2.real) ** 0.25)
        return complex(sqrt_r * cos(theta), sqrt_r * sin(theta))


# CUSTOMIZE by implementing new transforms

class Generator(object):
    def __init__(self, 
            ifs,
            instance="",
            seed = generate_seed(), 
            width = WIDTH, 
            height = HEIGHT, 
            iterations = ITERATIONS,
            num_points = NUM_POINTS,
            img_name = "test.png"):
        
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
        
        self.img = Image(self.width, self.height)
              
    def generate(self):
        random.seed(self.seed)
        for i in range(self.num_points):
            print "%s%i" % (self.instance, i)
            px = random.uniform(-1, 1)
            py = random.uniform(-1, 1)
            r, g, b = 0.0, 0.0, 0.0

            for j in range(self.iterations):
                t = self.ifs.choose()
                px, py = t.transform(px, py)
                r, g, b = t.transform_colour(r, g, b)

                fx, fy = self.ifs.final_transform(px, py)
                x = int((fx + 1) * self.width / 2)
                y = int((fy + 1) * self.height / 2)

                self.img.add_radiance(x, y, [r, g, b])  
                          
        self.img.save(self.img_name, max(1, (self.num_points * self.iterations) / (self.height * self.width)))   
             
    def __repr__(self):
        return "%s(\n%s,\nseed=%s, width=%s, height=%s, iterations=%s, num_points=%s, img_name='%s')" %(
            self.__class__.__name__,
            repr(self.ifs),
            self.seed,
            self.width,
            self.height,
            self.iterations,
            self.num_points,
            self.img_name)    
                
#CUSTOMIZE
TRANSFORM_CHOICES = [Linear, InverseJulia]

def generate_ifs():
    ifs = IFS()
    for n in range(NUM_TRANSFORMS):
        cls = random.choice(TRANSFORM_CHOICES)
        ifs.add(cls())
    return ifs
    
def read_ifs(ifs_file):
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", help="Input file containing an IFS generator description.  If no file is provided, an IFS generator is randomly constructed from TRANSFORM_CHOICES.", type=file)
    parser.add_argument("--desc", help="Output file where description of generated IFS is written.", type=argparse.FileType('w'), default="test.ifs")
    parser.add_argument("--image", help="Name of image file.", default="test.png")
    parser.add_argument("--instance", help="Instance identifier.")
    args = parser.parse_args()
     
    if args.gen:
        print 'Using generator file.'
        s = args.gen.read()
        g = eval(s)
    else:
        print 'Creating randomly selected generator.'
        g = Generator(ifs=generate_ifs(), img_name=args.image, instance=args.instance)
    g.generate()
    
    args.desc.write(repr(g))
 