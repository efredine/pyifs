import multiprocessing
import subprocess
import argparse

def work(cmd):
    return subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", help="Base file name for image and description files.", default="multi")
    args = parser.parse_args()
    
    count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=count)
    g = ('pypy pyifs.py --instance %i --image %s%i.png --desc %s%i.ifs' % (i,args.base,i,args.base,i) for i in range(count))
    print pool.map(work, g)
