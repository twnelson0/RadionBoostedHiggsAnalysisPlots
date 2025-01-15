# diff_tool.py

import difflib
import sys
import argparse

def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk))
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk))
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk))
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk))
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk))
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk))

def  color_diff ( diff ): 
    for  line  in  diff : 
        if  line . startswith ( '+' ): 
            yield "\033[92m {}\033[00m" .format(line) # Green 
        elif  line . startswith ( '-' ): 
            yield "\033[91m {}\033[00m" .format(line) # Red
        elif  line . startswith ( '^' ): 
            yield "\033[96m {}\033[00m" .format(line) # Cyan
        elif  line . startswith ( '@' ): 
            format = ';'.join([str(7), str(38), str(38)])
            yield '\x1b[%sm %s \x1b[0m' % (format, line)
        else : 
            yield line

def create_diff(old_file="", new_file =""):
    file_1 = open(old_file).readlines()
    file_2 = open(new_file).readlines()

    delta = difflib.unified_diff(file_1, file_2, fromfile=old_file, tofile=new_file)
    sys.stdout.writelines(color_diff(delta))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("old_file_version")
    parser.add_argument("new_file_version")
    args = parser.parse_args()

    old_file = args.old_file_version
    new_file = args.new_file_version

    create_diff(old_file, new_file)

if __name__ == "__main__":
    main()
