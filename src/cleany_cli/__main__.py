# Copyright (c) 2025 espehon
# MIT License

import sys
import os

# Support both direct execution and package module execution
if __name__ == "__main__":
    # Running directly: add parent directory to path for absolute imports
    src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, src_path)
    from cleany_cli.cleany import cleany
else:
    # Running as module (python -m cleany_cli): use relative import
    from .cleany import cleany


if __name__ == "__main__":
    cleany()
