import argparse
import os
import pyifs
import random
from ifsfiles import get_ifs_files

def get_child_name( p1, p2, i ):
    f1 = os.path.split(p1)[-1]
    f2 = os.path.split(p2)[-1]
    (name1, ext1) = os.path.splitext(f1)
    (name2, ext2) = os.path.splitext(f2)
    return '%s%s-%s' % (name1, name2, i)

def get_slice(l, i):
    if i > 0:
        random.shuffle(l)
        return l[:i]
    else:
        return []

def cross(l, args):
    for i in range(args.variants):
        random.shuffle(l)
        c = random.sample(l,args.parents)
        g = [pyifs.Generator.from_file(f) for f in c]
        tx = [x.ifs.transforms for x in g]
        child_name = "%s" % i
        t = []
        counts = [random.randint(1,len(x)-1) for x in tx]
        max_count = max([len(x) for x in tx])
        counts[-1] = max_count - sum(counts[:-1]) if max_count - sum(counts[:-1]) > 0 else 0
        t = []
        for x,count in zip(tx, counts):
            t += get_slice(x,count) 
        ifs = pyifs.IFS( t )
        random.shuffle(ifs.transforms)
        params = g[0].get_parameters()
        params['img_name'] = os.path.join(args.output_dir, child_name + '.png')
        child = pyifs.Generator(ifs, **params)
        
        child_file = open(os.path.join(args.output_dir, child_name + '.ifs'), 'w')
        child_file.write(repr(child))
        child_file.close()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("breed", help="Directory or file to breed.")
    parser.add_argument("--output_dir", help="Output directory for offspring specifications [.].", default=".")
    parser.add_argument("--variants", help="Number of variations to create [8].", type=int, default=8)
    parser.add_argument("--parents", help="Number of variations to create [2].", type=int, default=2)
    args = parser.parse_args()
    l = get_ifs_files(args.breed)
    
    cross(l, args)