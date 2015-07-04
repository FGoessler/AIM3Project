#!/usr/bin/env python
import sys, csv, itertools
rows = itertools.izip(*csv.reader(sys.stdin, delimiter=';'))
sys.stdout.writelines(';'.join(row) + '\n' for row in rows)