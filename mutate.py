import argparse
import os
import pyifs
import random
from ifsfiles import get_ifs_files

def get_mutant_name( p1, suffix, i ):
    f1 = os.path.split(p1)[-1]
    (name1, ext1) = os.path.splitext(f1)
    return '%s%s%i' % (name1, suffix, i)

def mutate(l1, args):
            
    for f1 in l1:
        for i in range(args.variants):
            g1 = pyifs.Generator.from_file(f1)
            g1.ifs.transforms = [(p, t.get_mutated_transform(args.percent) ) for (p,t) in g1.ifs.transforms]
            if args.before:
                g1.before = [pyifs.Linear()]
            else:
                g1.before = [t.get_mutated_transform(args.percent) for t in g1.before]
            if args.after:
                g1.after = [pyifs.Linear(params={'c':0.0, 'f':0.0})]
            else:
                g1.after = [t.get_mutated_transform(args.percent) for t in g1.after]
            mutant_name = get_mutant_name(f1, args.suffix, i)
            g1.img_name = os.path.join(args.output_dir, mutant_name + '.png')
            
            child_file = open(os.path.join(args.output_dir, mutant_name + '.ifs'), 'w')
            child_file.write(repr(g1))
            child_file.close()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="File or directory of files to mutate.")
    parser.add_argument("--percent", help="Percent of parameters to mutate out of 100.", default=50, type=int)
    parser.add_argument("--variants", help="Number of mutations per file.", default=4, type=int)
    parser.add_argument("--output_dir", help="Output directory for offspring specifications.", default=".")
    parser.add_argument("--suffix", help="Suffix to add to file name.", default="m")
    parser.add_argument("--before", help="Transform to put in before.")
    parser.add_argument("--after", help="Transform to put in after.")
    args = parser.parse_args()
    l1 = get_ifs_files(args.source)
    mutate(l1, args)
