import multiprocessing
import subprocess
import argparse
import os

def work(cmd):
    return subprocess.call(cmd, shell=True)

def get_base_name(args):
    
    target_dir = os.getcwd()
    
    if args.dir:
        if not os.path.exists(args.dir):
            print "Path name: '%s' does not exist.  Try creating it first." % args.dir
            exit(1)
        target_dir = args.dir
    
    if args.sub:
        target_dir = os.path.join(target_dir, args.sub)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

    return os.path.join(target_dir, args.base)

def run_jobs(g):
    count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=count)
    return pool.map(work, g)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Optional directory name to use (must exist).")
    parser.add_argument("--sub", help="Optional sub-directory to use - will be created if it doesn't exist.")
    parser.add_argument("--base", help="Base file name for image and description files.", default="multi")
    parser.add_argument("--runs", help="Number of images to generate.", default=8, type=int)
    args = parser.parse_args()
    
    base = get_base_name(args)
    
    g = ('pypy pyifs.py --instance %i --image %s%i.png --desc %s%i.ifs' % (i,base,i,base,i) for i in range(args.runs))
    print run_jobs(g)
