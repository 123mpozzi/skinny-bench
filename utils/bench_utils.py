import os
from pathlib import Path
from statistics import mean, pstdev

import xxhash


# Credit to https://stackoverflow.com/a/54477583
def hash_update_from_dir(directory, hash):
    assert Path(directory).is_dir(), f'Argument is not a directory: {directory}'
    for path in sorted(Path(directory).iterdir(), key=lambda p: str(p).lower()):
        hash.update(path.name.encode())
        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash.update(chunk)
        elif path.is_dir(): # it is recursive
            hash = hash_update_from_dir(path, hash)
    return hash

def hash_dir(directory: str) -> str:
    '''
    Return a hash hexdigest representing the directory

    Hash changes if filenames, filecontent, or number of files changes
    
    It is recursive
    '''
    hash = xxhash.xxh3_64()
    return hash_update_from_dir(directory, hash).hexdigest()

def hash_file(path: str) -> str:
    '''Return a hash hexdigest representing a file. Filename is not considered'''
    hash = xxhash.xxh3_64()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash.update(chunk)
    return hash.hexdigest()

def read_performance(perf_dir: str):
    '''Read inference time from performance benchmark files, and print it'''
    csv_sep = ','

    # will contain the final mean between each observation's mean
    observations_means = []

    # do the mean of each observation
    for i in range(5000):
        perf_filename = f'bench{i}.txt'
        perf_file = os.path.join(perf_dir, perf_filename)

        if not os.path.isfile(perf_file):
            break

        # read txt lines (as csv)
        file2c = open(perf_file)
        doubles = file2c.read().splitlines()
        file2c.close()

        intra_obs_timelist = []
        for entry in doubles: # ori_path, execution_time(s)
            ori_path = entry.split(csv_sep)[0]
            execution_time = entry.split(csv_sep)[1]
            intra_obs_timelist.append(float(execution_time))
        
        obs_mean = mean(intra_obs_timelist)
        obs_mean = '{:.6f}'.format(obs_mean) # round and zerofill
        obs_std = pstdev(intra_obs_timelist)
        obs_std = '{:.3f}'.format(obs_std) # round and zerofill

        obs_string = f'{obs_mean} ± {obs_std}'

        observations_means.append(obs_string)
        print(f'{perf_dir} at {i}: {obs_string}')
    
    # get the means from observation means, without the std
    obs_mean_values = []
    for entry in observations_means:
        obs_mean_values.append(float(entry.split(' ')[0]))
    
    # do the final mean of the observation means
    fin_mean = mean(obs_mean_values)
    fin_mean = '{:.6f}'.format(fin_mean) # round and zerofill
    fin_std = pstdev(obs_mean_values)
    fin_std = '{:.3f}'.format(fin_std) # round and zerofill

    fin_string = f'{fin_mean} ± {fin_std}'

    print(f'{perf_dir} at FIN: {fin_string}\n')
    return fin_string
