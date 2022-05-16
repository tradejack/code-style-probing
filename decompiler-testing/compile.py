#!/usr/bin/env python3.7
import py_compile
import decompyle3
import difflib

filename = "test"
source = filename+'.py'
target = "out/"+filename+"_out.py"

py_compile.compile(source)

with open(target, 'r+') as f:
    data = ""
    f.truncate(0)
    decompyle3.decompile_file('__pycache__/'+filename+'.cpython-38.pyc', outstream = f)
    f.seek(0)
    for line in f.readlines():
        if line[0] != '#':
            data += line
    f.truncate(0)
    f.seek(0)
    f.write(data)

diff = difflib.context_diff(open(source).readlines(), open(target).readlines(), n=0)
delta = ''.join(diff)
print(delta)