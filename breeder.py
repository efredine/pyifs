import argparse
import os
import pyifs
import random

def get_ifs_files(path):
    if not os.path.exists(path):
        print "Path: '%s' is not a valid file o directory." % path
        exit(1)
    
    if os.path.isfile(path):
        return [path]
        
    results = []
    for f in os.listdir(path):
        (file_name, ext) = os.path.splitext(f)
        if ext == '.ifs':
            results.append(os.path.join(path, f))
    
    if not len(results) > 0:
        print "Path: '%s' didn't contain any .ifs files." % path
        exit(1)
                
    return results

def get_child_name( p1, p2 ):
    f1 = os.path.split(p1)[-1]
    f2 = os.path.split(p2)[-1]
    (name1, ext1) = os.path.splitext(f1)
    (name2, ext2) = os.path.splitext(f2)
    return '%sx%s' % (name1, name2)

def cross_product(l1, l2, output_path):
    for f1 in l1:
        for f2 in l2:
            g1 = pyifs.Generator.from_file(f1)
            g2 = pyifs.Generator.from_file(f2)
            child_name = get_child_name(f1, f2)
            ifs = pyifs.IFS( g1.ifs.transforms + g2.ifs.transforms )
            random.shuffle(ifs.transforms)
            params = g1.get_parameters()
            del(params['seed'])
            params['img_name'] = child_name + '.png'
            child = pyifs.Generator(ifs, **params)
            
            child_file = open(os.path.join(output_path, child_name + '.ifs'), 'w')
            child_file.write(repr(child))
            child_file.close()
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("breed1", help="First directory or file to breed.")
    parser.add_argument("breed2", help="Second direcotry or file to breed.")
    parser.add_argument("--output_dir", help="Output directory for offspring specifications.", default=".")
    args = parser.parse_args()
    l1 = get_ifs_files(args.breed1)
    l2 = get_ifs_files(args.breed2)
    
    cross_product(l1, l2, args.output_dir)