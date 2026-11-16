"""Launch safely even when unrelated root scripts shadow Python standard modules."""
import sys
import os

project = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.curdir) != project]
sys.path.append(project)
from brainviz.__main__ import main

if __name__ == '__main__':
    raise SystemExit(main())
