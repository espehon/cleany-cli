# Copyright (c) 2025 espehon
# MIT License


import sys
from cleany import cleany

def main():
    raw_input = ' '.join(sys.argv[1:])
    sys.exit(cleany(raw_input))

if __name__ == "__main__":
    main()
