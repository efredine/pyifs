import argparse
from ifsfiles import get_ifs_files
from generate import run_jobs
import pyifs

def dump_it(g):
    for w,t in g.ifs.transforms:
        print repr(t)
        print t.initial_params

def update_files(ifs_files, iterations, points, scale, gamma, colour, zoom):
    for ifs_file in ifs_files:
        g = pyifs.Generator.from_file(ifs_file)  
        g.iterations = iterations
        g.num_points = points
        g.scale = scale
        g.gamma = gamma
        g.ifs.d = zoom
        if colour:
            g.ifs.transforms = [(p, t.get_mutated_colour() ) for (p,t) in g.ifs.transforms]
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
    
    args = parser.parse_args()
    
    ifs_files = get_ifs_files(args.gen)
    update_files( ifs_files, args.iterations, args.points, args.scale, args.gamma, args.colour, args.zoom )
    run_jobs(('pypy pyifs.py --gen %s  --desc %s' % (f, f) for f in ifs_files))