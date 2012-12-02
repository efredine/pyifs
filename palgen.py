import re
import sys
import os
from palette import Palette

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv/3], 16) for i in range(0, lv, lv/3))

def rgb_to_hex(rgb):
    rgb = eval(rgb)
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    return '#%02X%02X%02X' % (r,g,b)
    
def get_color_tuple(color):
    if re.search('\#[a-fA-F0-9][a-fA-F0-9][a-fA-F0-9][a-fA-F0-9][a-fA-F0-9][a-fA-F0-9]', color):
        return hex_to_rgb(color)
    elif re.search('[0-9]{1,3}, [0-9]{1,3}, [0-9]{1,3}', color):
        return rgb_to_hex(color)
    else:
        return ()
        
def get_theme(color_list):
    t = [get_color_tuple(color) for color in color_list]
    return [(r/255.0, g/255.0, b/255.0) for (r,g,b) in t]
    
def get_file_name(path):
    p = os.path.split(path)
    f = p[-1]
    d = os.path.join(*p[:-1])
    (name,ext) = os.path.splitext(f)
    return os.path.join(d, name+'.ptl')
    
def main():
    for line in sys.stdin:
        path, color_list = line.split()
        theme = get_theme(color_list.split(','))
        p = Palette(*theme)
        p_file_name = get_file_name(path)
        sys.stderr.write(p_file_name + '\n')
        p_file = open(p_file_name, 'w')
        p_file.write(repr(p))
        p_file.close()
                
if __name__ == '__main__':
  main()