import os

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
