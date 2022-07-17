#!/usr/bin/python3

import os

def get_total_dir_size(directory: str) -> int:
    '''
    Returns the number of files inside the given directory excluding directories themself in the final sum. Working recursively.
    
    :param directory: Directory path, that should be scanned
    :return: Number of files in the given directory
    '''
    return sum([1 if x.is_file() else get_total_dir_size(f"{directory}/{x.name}") for x in os.scandir(directory)])