import random
import sys
import argparse
from math import cos, sin, tan, pi, atan, atan2, sqrt, exp, isnan
import cmath
from datetime import datetime

from image import Image, SCALEFACTOR_ADJUST, GAMMA_ADJUST
import colorsys
from palette import Palette

# CUSTOMIZE
WIDTH = 1024
HEIGHT = 1024
ITERATIONS = 10000
NUM_POINTS = 1000

NUM_TRANSFORMS =   17

def random_complex():
    return complex(random.uniform(-1, 1), random.uniform(-1, 1))
    
def generate_seed():
    # now = datetime.utcnow() - datetime.utcfromtimestamp(0)
    # return (now.days, now.seconds, now.microseconds)
    return random.SystemRandom().random()

class IFS:
    
    def __init__(self, transforms = [], a=1.0, b=0.0, c=0.0, d=0.5, rotate=0.0, origin=(0.0, 0.0)):
        self.transforms = transforms
        self.total_weight = sum([x for (x,y) in transforms])
        self.rotate=rotate
        self.origin=origin
        self.origin_complex = complex(origin[0], origin[1])
        
        #final Mondrian transform parameters
        self.a = a # 0.5 (provides a zoom/scale tranform)
        self.b = b # 0
        self.c = c # 0 
        self.d = d # 1.0
    
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
        z = complex(px, py) + self.origin_complex
        z2 = (self.a * z + self.b) / (self.c * z + self.d)
        if self.rotate != 0.0:
            (r,phi) = cmath.polar(z2)
            phi += self.rotate
            z2 = cmath.rect(r,phi)
        return z2.real, z2.imag              


    def config_parameters(self):
        return "a=%s, b=%s, c=%s, d=%s, rotate=%s, origin=%s" % (
            repr(self.a),
            repr(self.b),
            repr(self.c),
            repr(self.d),
            repr(self.rotate),
            repr(self.origin)
        )
        
    def __repr__(self):
        return "%s([\n%s], \n%s)" % (self.__class__.__name__, 
                                ',\n'.join(["\t(%s,%s)" % (x, repr(y)) for (x,y) in self.transforms]),
                                self.config_parameters() 
        )


class Transform(object):
    
    def __init__(self, params={}):
        self.params = {}
        self.initial_params = params
        self.seteither('palette', params, Palette())
        red, green, blue = self.palette.get_rgb_choice()
        self.seteither('red', params, red)
        self.seteither('green', params, green)
        self.seteither('blue', params, blue)

        # if random.choice([True,False]):
        #     h = random.uniform(0.0/360, 60.0/360)
        # else:
        #     h = random.uniform(180.0/360, 240.0/360)
        # h = random.uniform(0.0,1.0)
        # s = random.uniform(0.0,1.0)
        # l = random.uniform(0.0,1.0)
        # r,g,b = colorsys.hls_to_rgb(h,l,s)
        # h = random.random()
        # r,g,b = colorsys.hls_to_rgb(h,l,s)
        
        
    
    def seteither(self, name, params, value):
        if params.has_key(name):
            self.params[name] = params[name]
        else:
            self.params[name] = value
        setattr(self, name, self.params[name])
    
    def transform_colour(self, r, g, b):
        r = (self.red + r) / 2 
        g = (self.green + g) / 2 
        b = (self.blue + b) / 2 
        return r, g, b
        
    def get_new_class(self, params):
        return self.__class__(params, **self.other_parameters())
        
    def get_new_transform(self, params={}):
        return self.get_new_class(params=dict(self.initial_params.items() + params.items()))
        
    def get_mutated_transform(self, percent):
        if percent == 100:
            params = self.params
        elif percent == 0:
            params = {}
        else:
            items = self.params.items()
            random.shuffle( items )
            keep = min(percent * len(items) / 100, len(items))
            params = dict( items[0:keep if keep > 0 else 0] )
            # print "Keeping %s of %s params for %s." % (keep, len(items), self.__class__.__name__)        
        return self.get_new_class(params=params)
        
    def get_mutated_colour(self, palette=None):
        params = self.params
        del(params['red'])
        del(params['green'])
        del(params['blue'])
        if palette:
            params['palette'] = palette
        return self.__class__(params=params)
        
    def get_colour(self):
        return self.params['red'], self.params['green'], self.params['blue']
        
    def set_colour(self, r, g, b):
        self.params['red'] = r
        self.params['green'] = g
        self.params['blue'] = b
        
    def other_parameters(self):
        return {}

    def __repr__(self):
        d = self.other_parameters()
        others = ["%s=%s" % (k, repr(v)) for k,v in d.items()]
        return "%s(params=%s, %s)" % (self.__class__.__name__, repr(self.params), ','.join(others));

# Linear transformations
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


class Rotate(Transform):
    def __init__(self, params={}, rotate_range=(0,2*pi)):
        super(Rotate, self).__init__(params)
        self.rotate_range = rotate_range
        self.seteither('rotate', params, random.uniform(self.rotate_range[0], self.rotate_range[1]))
    
    def get_new_class(self, params):
        return self.__class__(params, rotate_range=self.rotate_range)
             
    def transform(self, x, y):
        theta = atan2(y,x)
        r = sqrt(x*x + y*y)
        theta += self.rotate
        return r * cos(theta), r * sin(theta)

    def other_parameters(self):
        d = super(Rotate, self).other_parameters()
        d['rotate_range'] = self.rotate_range
        return d

class Translate(Transform):
    def __init__(self, params={}):
        super(Translate, self).__init__(params)
        self.seteither('move_x', params, random.uniform(-1,1))
        self.seteither('move_y', params, random.uniform(-1,1))

    def transform(self, x, y):
        return x + self.move_x, y + self.move_y

class Scale(Transform):
    def __init__(self, params={}, scale_range=(-1, 1)):
        super(Scale, self).__init__(params)
        self.scale_range = scale_range
        self.seteither('scale', params, random.uniform(self.scale_range[0],self.scale_range[1]))
        
    def get_new_class(self, params):
        return self.__class__(params, scale_range=self.scale_range)

    def transform(self, x, y):
        return self.scale * x, self.scale * y

    def other_parameters(self):
        d = super(Scale, self).other_parameters()
        d['scale_range'] = self.scale_range
        return d

class Flip(Transform):
    def __init__(self, params={}):
        super(Flip, self).__init__(params)
        self.seteither('flip_x', params, random.uniform(-1,1))
        self.seteither('flip_y', params, random.uniform(-1,1))

    def transform(self, x, y):
        return self.flip_x * x, self.flip_y * y


# Complex transformations
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

# Functions
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
        return x/r2 if r2 > 0 else 1, y/r2 if r2 > 0 else 1

class Sinusoidal(Transform):

    def transform(self, x, y):
        return sin(x), sin(y)
        
class Tangent(Transform):
    
    def transform(self, x, y):
        return sin(x)/cos(y), tan(y)
        
class Swirl(Transform):

    def transform(self, x, y):
        r2 = x * x + y * y
        return x * sin(r2) - y * cos(r2), x * cos(r2) + y * sin(r2)

class HorseShoe(Transform):
 
    def transform(self, x, y):
        r = sqrt(x*x + y*y)
        d1 = r * ((x-y)*(x+y))
        d2 = r * 2*x*y
        return 1/d1 if d1 > 0 else 1, 1/d2 if d2 > 0 else 1
        
class Polar(Transform):
    
    def transform(self, x, y):
        theta = atan2(y,x)
        r = sqrt(x*x + y*y)
        return theta, (r - 1)
        
class Handkerchief(Transform):
    
    def transform(self, x, y):
        theta = atan2(y,x)
        r = sqrt(x*x + y*y)
        return r * sin(theta + r), r * cos(theta - r)
        
class Heart(Transform):
    
    def transform(self, x, y):
        theta = atan2(y,x)
        r = sqrt(x*x + y*y)
        return r * sin(theta * r), -1 * r * cos(theta * r)
        
class Disc(Transform):

    def transform(self, x, y):
        theta = atan2(y,x)
        r = sqrt(x*x + y*y)
        return theta * sin(pi * r), theta * cos(pi * r)
        
class Spiral(Transform):
    
    def transform(self, x, y):
        theta = atan2(y,x)
        r = sqrt(x*x + y*y)
        return 1/r * (cos(theta) + sin(r)), 1/(r if r > 0 else 1) *(sin(theta) - cos(r))

class Hyperbolic(Transform):
    
    def transform(self, x, y):
        theta = atan2(y,x)
        r = sqrt(x*x + y*y)
        return  sin(theta)/r if r > 0 else 1, r * cos(theta)
        

class Diamond(Transform):
    
    def transform(self, x, y):
        theta = atan2(y,x)
        r = sqrt(x*x + y*y)
        return  2 * sin(theta) * cos(r), 2 * cos(theta) * sin(r)    

class Ex(Transform):
    
    def transform(self, x, y):
        theta = atan2(y,x)
        r = sqrt(x*x + y*y)
        p0 = sin(theta+r)
        p1 = cos(theta-r)
        return  r *(p0*p0*p0 + p1*p1*p1), r*(p0*p0*p0 - p1*p1*p1)
        
class Fisheye(Transform):
    
    def transform(self, x, y):
        r = sqrt(x*x + y*y)
        c = 2/(r+1)
        return c*y, c*x
        
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

class BentParams(Transform):

    def __init__(self, params={}):
        super(BentParams, self).__init__(params)
        self.seteither('bend', params, random.uniform(1,3)) 

    def transform(self, x, y):
        if x >= 0 and y >= 0:
             return x,y
        elif x < 0 and y >= 0:
            return self.bend*x, y
        elif x >= 0 and y < 0:
            return x, y/self.bend
        else:
            return self.bend*x, y/self.bend
          
class Cross(Transform):
    
    def transform(self, x, y):
        s = (x**2 - y**2)**2
        s = s if s > 0 else 1
        s = sqrt(1/s)
        return s * x, s * y

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

class Curl(Transform):

    def __init__(self, params={}):
        super(Curl, self).__init__(params)
        self.seteither('p1', params, random.random()) 
        self.seteither('p2', params, random.random())

    def transform(self, x, y):
        t1 = 1 + self.p1 * x + self.p2*(x*x - y*y)
        t2 = self.p1*y * 2*self.p2*x*y
        c = 1/(t1*t1 + t2*t2)
        return c * (x*t1 + y*t2), c*(y*t1 - x*t2)


class Rectangle(Transform):

    def __init__(self, params={}):
        super(Rectangle, self).__init__(params)
        self.seteither('p1', params, random.uniform(-1, 1)) 
        self.seteither('p2', params, random.uniform(-1, 1))        

    def transform(self, x, y):
        return (2*int(x/self.p1) + 1)*self.p1 - x, (2*int(y/self.p2) + 1)*self.p2 - y     

class RectangleSkip(Transform):

    def __init__(self, params={}):
        super(RectangleSkip, self).__init__(params)
        self.seteither('p1', params, random.uniform(-1, 1)) 
        self.seteither('p2', params, random.uniform(-1, 1))     
        self.seteither('x_skip', params, random.randint(1,3)) 
        self.seteither('y_skip', params, random.randint(1,3))
        
    def transform(self, x, y):
        return (self.x_skip*int(x/self.p1) + 1)*self.p1 - x, (self.y_skip*int(y/self.p2) + 1)*self.p2 - y     

    
class Rays(Transform):

    def __init__(self, params={}):
        super(Rays, self).__init__(params)
        self.seteither('rays', params, random.random()) 

    def transform(self, x, y):
        r2 = x*x + y*y
        coefficient = self.rays * tan( random.random()*pi*self.rays ) / (r2 if r2 > 0 else 1.0)
        coefficient = 10.0 if isnan(coefficient) or coefficient > 10.0 else coefficient
        coefficient = -10.0 if coefficient < -10.0 else coefficient
        return  coefficient * cos(x), coefficient * sin(y)

class Blade(Transform):

    def __init__(self, params={}):
        super(Blade, self).__init__(params)
        self.seteither('blade', params, random.random()) 

    def transform(self, x, y):
        r = sqrt(x*x + y*y)
        p = random.random()
        return x * ( cos(p*r*self.blade) + sin(p*r*self.blade) ), x * (cos(p*r*self.blade) - sin(p*r*self.blade))

class MadeUp(Transform):

    def __init__(self, params={}):
        super(MadeUp, self).__init__(params)
        self.seteither('madeup', params, random.random()) 

    def transform(self, x, y):
        r = sqrt(x*x + y*y)
        return x * ( cos(r*self.madeup) + sin(r*self.madeup) ), x * (cos(r*self.madeup) - sin(r*self.madeup))
        
class PDJ(Transform):
    
    def __init__(self, params={}):
         super(PDJ, self).__init__(params)
         self.seteither('pdj_p1', params, random.uniform(-1, 1)) 
         self.seteither('pdj_p2', params, random.uniform(-1, 1)) 
         self.seteither('pdj_p3', params, random.uniform(-1, 1)) 
         self.seteither('pdj_p4', params, random.uniform(-1, 1)) 

    def transform(self, x, y):
        return sin(self.pdj_p1 * y) - cos(self.pdj_p2 * x), sin(self.pdj_p3 * x) - cos(self.pdj_p4 * y)
        
class Waves(Transform):
    
    def __init__(self, params={}):
         super(Waves, self).__init__(params)
         self.seteither('wave_b', params, random.uniform(-1, 1)) 
         self.seteither('wave_c', params, random.uniform(-1, 1)) 
         self.seteither('wave_e', params, random.uniform(-1, 1)) 
         self.seteither('wave_f', params, random.uniform(-1, 1)) 

    def transform(self, x, y):
        return x + self.wave_b * sin(y/self.wave_c**2), y + self.wave_e * sin(x/self.wave_f**2)

#
class Fan2(Transform):
    
    def __init__(self, params={}):
         super(Fan2, self).__init__(params)
         p1 = pi * (random.random())**2
         self.seteither('fan_p1', params, p1 if p1 > 0 else 1)
         self.seteither('fan_p2', params, random.random()) 

    def transform(self, x, y):
        theta = atan2(y,x)
        r = sqrt(x*x + y*y)
        t = theta + self.fan_p2 - self.fan_p1 * int(2*theta*self.fan_p2/self.fan_p1)
        
        if t > self.fan_p1/2:
            return r * sin(theta - self.fan_p1/2), r * cos(theta - self.fan_p1/2)
        else:
            return r * sin(theta + self.fan_p1/2), r * sin(theta + self.fan_p1/2)
    
 
# Noise injection
class RandomMove(Transform):
    
    def transform(self, x, y):
        return random.uniform(-2, 2), random.uniform(-2, 2)

class Square(Transform):

    def transform(self, x, y):
        return random.random() - 0.5, random.random() - 0.5

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
    def __init__(self, params={}, colour_weight=1.0):
        super(Sequence, self).__init__(params)
        self.colour_weight = colour_weight
        sequence = [(prob, t.get_new_transform()) for (prob, t) in params['sequence']]
        self.params['sequence'] = sequence
        setattr(self, 'sequence', sequence)
                
    def transform(self, px, py):
        for (prob, instance) in self.sequence:
            if prob > random.random():
                px,py = instance.transform(px,py)
        return px,py

    def get_mutated_transform(self, percent):
        new_sequence = []
        for (prob, instance) in self.sequence:
            new_sequence.append( (prob, instance.get_mutated_transform(percent)) )
        return self.__class__(params={'sequence': new_sequence})

    def transform_colour(self, r, g, b):
        r = (self.red + r) / 2 * self.colour_weight
        g = (self.green + g) / 2 * self.colour_weight
        b = (self.blue + b) / 2  * self.colour_weight
        return r, g, b

    def other_parameters(self):
        d = super(Sequence, self).other_parameters()
        d['colour_weight'] = self.colour_weight
        return d
 
                        
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
            scale=SCALEFACTOR_ADJUST,
            gamma=GAMMA_ADJUST,
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
        self.scale = scale
        self.gamma = gamma
        self.iterations = iterations
        self.num_points = num_points
        self.img_name =  img_name  
        self.before = before
        self.after = after
        
        self.img = Image(self.width, self.height, self.scale, self.gamma)
              
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

                if j > 20:
                    self.img.add_radiance(x, y, [r, g, b])  
                          
        self.img.save(self.img_name, max(1, (self.num_points * self.iterations) / (self.height * self.width)))   
             
    def __repr__(self):
        return "%s(\n%s,\nseed=%s, width=%s, height=%s, scale=%s, gamma=%s, iterations=%s, num_points=%s, img_name='%s', before=%s, after=%s)" %(
            self.__class__.__name__,
            repr(self.ifs),
            self.seed,
            self.width,
            self.height,
            self.scale,
            self.gamma,
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
            'scale': self.scale,
            'gamma': self.gamma,
            'iterations': self.iterations,
            'num_points': self.num_points,
            'img_name': self.img_name,
            'before': self.before,
            'after': self.after
        }
         
ALL_TRANSFORMS = [LinearCenter(), Linear(), Moebius(), InverseJulia(), Identity(), Swap(), 
    Spherical(), Sinusoidal(), Swirl(), HorseShoe(), Polar(), Handkerchief(), Heart(), Disc(), Spiral(), Hyperbolic(), 
    Diamond(), Ex(), Bent(), Perspective(), Rectangle(), Curl(), Rays(), Blade(), Waves(), Fisheye(), PDJ(), Cross()]    
    
NOISE = [RandomMove(), Wiener(), RandomWalk(), Square()]                                

MODIFIED = Sequence(params={'sequence':[(1.0, Rectangle()), (1.0, Scale()), (1.0, Translate())]})
MODIFIED2 = Sequence(params={'sequence':[(1.0, Fan2()), (1.0, Translate())]})
MODIFIED3 = Sequence(params={'sequence':[(1.0, Spherical()), (1.0, Translate()), (1.0, Rotate()), (1.0, Scale())]})
MODIFIED4 = Sequence(params={'sequence':[(1.0, Curl()), (1.0, Translate()), (1.0, Rotate()), (1.0, Scale())]})

SINE_MOD = Sequence(params={'sequence':[(1.0, Sinusoidal()), (1.0, Rotate(params={'rotate': 2 * pi * 45 / 60}))]})
SINE_MOD2 = Sequence(params={'sequence':[(1.0, Sinusoidal()), (1.0, Translate(params={'move_x': -0.5, 'move_y': -0.5}))]})
SINE_MOD3 = Sequence(params={'sequence':[(1.0, Sinusoidal()), (1.0, Flip(params={'flip_x': 0.5, 'flip_y': 0.5}))]})

# TRANSFORM_CHOICES = [Sequence(params={'sequence':[(1.0, Linear()), (1.0, Disc()), (1.0, Perspective())]}), InverseJulia()]
# TRANSFORM_CHOICES = [PDJ(), Moebius()]
# TRANSFORM_CHOICES = [MODIFIED, Moebius()]
# TRANSFORM_CHOICES = ALL_TRANSFORMS
# TRANSFORM_CHOICES = [Linear(), Cross(), MODIFIED, Rectangle(), Bent(), MODIFIED2]
# TRANSFORM_CHOICES = [Linear(), Fan2()]
# TRANSFORM_CHOICES = [MODIFIED3, MODIFIED4, MODIFIED2, Wiener()]
MODIFIED5 = Sequence(params={'sequence':[(1.0, Blade()), (1.0, Translate()), (1.0, Rotate()), (1.0, Scale())]})

CHOICE1 = Sequence(params={'sequence':[(1.0, Rectangle()), (1.0, MODIFIED), (1.0, MODIFIED)]})
CHOICE1B = Sequence(params={'sequence':[(1.0, MODIFIED), (1.0, MODIFIED)]})
CHOICE2 = Sequence(params={'sequence':[(1.0, Rectangle()), (1.0, MODIFIED)]})
CHOICE3 = Sequence(params={'sequence':[(1.0, Rectangle())]})

VERTICAL = Linear(params={'a': 0.0, 'b': 0.0, 'd': 0.0})
HORIZONTAL = Linear(params={'b': 0.0, 'd': 0.0, 'e': 0})
VRECT = Sequence(params={'sequence':[(1.0, VERTICAL), (1.0, Rectangle())]})
HRECT = Sequence(params={'sequence':[(1.0, HORIZONTAL), (1.0, Rectangle())]})
STEEP1 = Sequence(params={'sequence':[(1.0, VERTICAL), (1.0, Rotate(params={'rotate': 0.5/360*2*pi}))]})
STEEP2 = Sequence(params={'sequence':[(1.0, VERTICAL), (1.0, Rotate(params={'rotate': 1.0/360*2*pi}))]})
STEEP3 = Sequence(params={'sequence':[(1.0, VERTICAL), (1.0, Rotate(params={'rotate': 1.5/360*2*pi}))]})
STEEP4 = Sequence(params={'sequence':[(1.0, VERTICAL), (1.0, Rotate(params={'rotate': 2.0/360*2*pi}))]})
STEEP = Sequence(colour_weight=0.01, 
    params={'sequence':
            [(1.0, VERTICAL), (1.0, Rotate(rotate_range=(0.5/360*2*pi, 2.5/360*2*pi))), (1.0, Scale(scale_range=(0.05,0.6)))],
        'red': 0.75,
        'green': 0.75,
        'blue': 0.75
    })
# STEEP = Sequence(params={'sequence':[(1.0, VERTICAL), (1.0, Rotate())]})
RayMod = Sequence(colour_weight=0.75, params={'sequence':[(1.0, Rays()), (1.0, Rotate())]})
# TRANSFORM_CHOICES = [STEEP, STEEP, STEEP, CHOICE1, CHOICE1B, CHOICE2, CHOICE3,  CHOICE1, CHOICE1B, CHOICE2, CHOICE3]

M1 = Sequence(params={'sequence':[(1.0, PDJ()), (1.0, Scale()), (1.0, Translate())]})

STEEP5 = Sequence(#colour_weight=0.1,
    params={'sequence':
            [(1.0, VERTICAL), (1.0, Rotate(rotate_range=(-2.5/360*2*pi, 2.5/360*2*pi))), (1.0, Scale(scale_range=(0.05,0.6)))]
        # 'red': 0.75,
        # 'green': 0.75,
        # 'blue': 0.75
    })
#TRANSFORM_CHOICES = [STEEP5, STEEP5, STEEP5, CHOICE1, CHOICE1B, CHOICE2, CHOICE3,  CHOICE1, CHOICE1B, CHOICE2, CHOICE3, Swap(), Swap()]

SLAT = Sequence(
    params={'sequence':
            [(1.0, HORIZONTAL), (1.0, Rotate(rotate_range=(-21.0/360*2*pi, -23.0/360*2*pi))), (1.0, Scale(scale_range=(0.05,0.6)))]
        # 'red': 0.25,
        # 'green': 0.25,
        # 'blue': 0.25
    })

#TRANSFORM_CHOICES = [SLAT, SLAT, STEEP5, STEEP5, CHOICE1B, CHOICE2, CHOICE3, CHOICE1B, CHOICE2, CHOICE3, CHOICE1B, CHOICE2, CHOICE3, CHOICE1B, CHOICE2, CHOICE3]
HS = Sequence(params={'sequence':[(1.0, Rectangle()), (1.0, Scale()), (1.0, Translate()), (1.0, Rotate(rotate_range=(-11.0/360*2*pi, 11.0/360*2*pi)))]})
HS2 = Sequence(params={'sequence':[(1.0, HS), (1.0, HS)]})

# TRANSFORM_CHOICES = [STEEP5, HS, HS, HS, HS2, Swap()]
XXY = Sequence(params={'sequence':[(1.0, InverseJulia()), (1.0, Scale(scale_range=(0.1, 1.0)))]})
YYW = Sequence(params={'sequence':[(1.0, Waves()), (1.0, Scale(scale_range=(0.1, 0.3))), (1.0, Translate())]})
YYX = Sequence(params={'sequence':[(1.0, Curl()), (1.0, Scale(scale_range=(0.1, 1.0))), (1.0, Translate())]})
YYY = Sequence(params={'sequence':[(1.0, Sinusoidal()), (1.0, Scale(scale_range=(0.1, 1.0))), (1.0, Translate())]})
YYZ = Sequence(params={'sequence':[(1.0, Spherical()), (1.0, Scale(scale_range=(0.1, 1.0))), (1.0, Translate())]})
XYZ = Sequence(params={'sequence':[(1.0, YYZ), (1.0, Disc())]})
ZYX = Sequence(params={'sequence':[(1.0, XXY), (1.0, Waves()), (1.0, Translate())]})
# TRANSFORM_CHOICES = [CHOICE1, CHOICE1B, CHOICE2, CHOICE3, YYW, YYX, YYY, YYZ]
# TRANSFORM_CHOICES = [VERTICAL, VERTICAL, VERTICAL, CHOICE2, YYW, YYW, YYY, YYY]

# XXY = Sequence(params={'sequence':[(1.0, Spherical()), (1.0, Scale(scale_range=(0.25, 0.50)))]})
# XYZ = Sequence(params={'sequence':[(1.0, Perspective()), (1.0, XXY), (1.0, Translate())]})
# X1 = Sequence(params={'sequence':[(1.0, Disc()), (1.0, Translate())]})
# X2 = Sequence(params={'sequence':[(1.0, Curl()), (1.0, Translate())]})
# ATT_WAVE = Sequence( colour_weight=0.1,
#     params={'sequence':
#             [(1.0, Waves())]
#         # 'red': 0.25,
#         # 'green': 0.25,
#         # 'blue': 0.25
#     })

SKIPMOD = Sequence(params={'sequence':[(1.0, RectangleSkip()), (1.0, Scale()), (1.0, Translate())]})
SKIP1 = Sequence(params={'sequence':[(1.0, RectangleSkip()), (1.0, SKIPMOD), (1.0, SKIPMOD)]})
SKIP2 = Sequence(params={'sequence':[(1.0, SKIPMOD), (1.0, SKIPMOD)]})
SKIP3 = Sequence(params={'sequence':[(1.0, RectangleSkip()), (1.0, SKIPMOD)]})

BPX = Sequence(params={'sequence':[(1.0, BentParams()), (1.0, Translate())]})
TRANSFORM_CHOICES = [RectangleSkip(), SKIP1, SKIP2, SKIP3, BentParams(), BentParams()]


def generate_ifs():
    ifs = IFS()
    for n in range(NUM_TRANSFORMS):
        transform = random.choice(TRANSFORM_CHOICES)
        ifs.add(transform.get_new_transform())
    return ifs
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", help="Input file containing an IFS generator description.  If no file is provided, an IFS generator is randomly constructed from TRANSFORM_CHOICES.", type=file)
    parser.add_argument("--desc", help="Output file where description of generated IFS is written.", default="test.ifs")
    parser.add_argument("--image", help="Name of image file.", default="test.png")
    parser.add_argument("--instance", help="Instance identifier.")
    args = parser.parse_args()
     
    if args.gen:
        print 'Using generator file.'
        g = Generator.from_open_file(args.gen)
    else:
        print 'Creating randomly selected generator.'
        ifs = generate_ifs()
        # ifs = IFS()
        # ifs.add(Moebius())
        # ifs.add(Rectangle())
        # ifs.add(InverseJulia())
        # ifs.add(Spherical())
        # ifs.add(Sinusoidal())
        # ifs.add(Linear())
        # ifs.add(RandomMove())
        g = Generator(ifs=ifs, img_name=args.image, instance=args.instance)#, after=[Perspective(params={'p1': 33.0/360.0*2*pi, 'p2': 0.1})])
    g.generate()
    
    desc_file = open(args.desc, 'w')
    desc_file.write(repr(g))
 