import argparse
from ifsfiles import get_ifs_files
from generate import run_jobs
import pyifs

def dump_it(g):
    for w,t in g.ifs.transforms:
        print repr(t)
        print t.initial_params

def update_files(ifs_files, iterations, points):
    for ifs_file in ifs_files:
        g = pyifs.Generator.from_file(ifs_file)  
        g.iterations = iterations
        g.num_points = points
        f = open(ifs_file, 'w')
        f.write(repr(g))
        f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("gen", help="Directory or file to regenerate.")
    parser.add_argument("--iterations", help="Number of iterations.", default=20000, type=int)
    parser.add_argument("--points", help="Number of points.", default=2000, type=int)
    
    args = parser.parse_args()
    
    ifs_files = get_ifs_files(args.gen)
    update_files( ifs_files, args.iterations, args.points )
    run_jobs(('pypy pyifs.py --gen %s' % (f) for f in ifs_files))