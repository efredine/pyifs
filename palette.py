import random
import colorsys

class Palette:

    @classmethod
    def from_open_file(cls, file_handle):
        s = file_handle.read()
        return eval(s)
        
    @classmethod
    def from_file(cls, file_name):
        f = open(file_name)
        p = Palette.from_open_file(f)
        f.close()
        return p
    
    def __init__(self, *themes):
        self.themes = themes
        
    def get_rgb_choice(self, saturate=False):
        if self.themes:
            r,g,b = random.choice(self.themes)
        else:
            r,g,b = (random.random(), random.random(), random.random())
        
        if saturate:
            h,s,v = colorsys.rgb_to_hsv(r,g,b)
            r,g,b = colorsys.hsv_to_rgb(h,0.8,v)
            
        return r,g,b
            
    def __repr__(self):
        if not self.themes:
            return "%s()" % self.__class__.__name__
        else:
            return "%s(*%s)" % (self.__class__.__name__, repr(self.themes))
    