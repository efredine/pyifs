import argparse
from ifsfiles import get_ifs_files
from generate import run_jobs
import pyifs
from palette import Palette
import colorsys

def adjust_saturation(g, saturation):
    for (p,t) in g.ifs.transforms:
        r,g,b = t.get_colour()
        h,s,v = colorsys.rgb_to_hsv(r,g,b)
        s = s + saturation if s + saturation < 1.0 else 1.0
        r,g,b = colorsys.hsv_to_rgb(h,s,v)
        t.set_colour(r,g,b)    

def dump_it(g):
    for w,t in g.ifs.transforms:
        print repr(t)
        print t.initial_params

def update_files(ifs_files, args):
    palette = None
    if args.palette:
        palette = Palette.from_open_file(args.palette)
    
    for ifs_file in ifs_files:
        g = pyifs.Generator.from_file(ifs_file)  
        if args.iterations:
            g.iterations = args.iterations
        if args.points:
            g.num_points = args.points
        if args.scale:
            g.scale = args.scale
        if args.gamma:
            g.gamma = args.gamma
        if args.zoom:
            g.ifs.d = args.zoom
        if args.colour or palette:
            g.ifs.transforms = [(p, t.get_mutated_colour(palette=palette) ) for (p,t) in g.ifs.transforms]
        if args.saturation:
            adjust_saturation(g, args.saturation)
        f = open(ifs_file, 'w')
        f.write(repr(g))
        f.truncate()
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("gen", help="Directory or file to regenerate.")
    parser.add_argument("--iterations", help="Number of iterations.", default=10000, type=int)
    parser.add_argument("--points", help="Number of points.", default=1000, type=int)
    parser.add_argument("--scale", help="Lighting scale adjustment.", default=1.0, type=float)
    parser.add_argument("--gamma", help="Gamma adjustment.", default=0.00, type = float)
    parser.add_argument("--zoom", help="Camera zoom - bigger numbers zoom out, default is 0.5.", default=0.5, type=float)
    parser.add_argument("--colour", help="Mutate colours.", action="store_true")
    parser.add_argument("--palette", help="Palette file.", type=file)
    parser.add_argument("--saturation", help="Saturation adjustment.", type=float)
    
    
    args = parser.parse_args()
    
    ifs_files = get_ifs_files(args.gen)
    update_files( ifs_files, args)
    run_jobs(('pypy pyifs.py --gen %s  --desc %s' % (f, f) for f in ifs_files))