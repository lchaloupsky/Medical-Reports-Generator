import os

def get_total_dir_size(directory: str) -> int:
    '''Returns the number of files inside the given directory excluding directories themself in the final sum. Working recursively.'''
    return sum([1 if x.is_file() else get_total_dir_size(f"{directory}/{x.name}") for x in os.scandir(directory)])