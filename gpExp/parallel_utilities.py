#Copyright (c) 2013-2015, Massachusetts Institute of Technology
#
#This file is part of GPEXP:
#Author: Alex Gorodetsky goroda@mit.edu
#
#GPEXP is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 2 of the License, or
#(at your option) any later version.
#
#GPEXP is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with GPEXP.  If not, see <http://www.gnu.org/licenses/>.

#Code


import multiprocessing as mp
import numpy as np

def parallelizeMcForLoop(func, fargs, arrayToSplit):
        
    def worker(nums, out_q, fff):
            """ The worker function, invoked in a process. 'nums' is an array?
             The results are placed in
             a dictionary that's pushed to a queue.
            """
            #print "Process: ", mp.current_process().name
            outarray = func(nums, fff)
            outdict = {}
            outdict[int(mp.current_process().name)]=outarray
            out_q.put(outdict)
            
        # Each process will get 'chunksize' nums and a queue to put his out
        # dict into
    
    nPoints = arrayToSplit.shape[0]
    out_q = mp.Queue()
    nprocs = mp.cpu_count()
    #print "Parallelizing into ", nprocs, " processors"
    chunksize = int(np.ceil(nPoints)/ float(nprocs))
    procs = []
    for i in range(nprocs-1):
        #print "range: ", chunksize*i, chunksize*(i+1)
        p = mp.Process(
                target=worker,
                args=(arrayToSplit[chunksize * i:chunksize * (i + 1),:],
                      out_q, fargs), name=str(i))
        procs.append(p)
        p.start()
    # last one
    #print "range: ", chunksize*(nprocs-1)
    p = mp.Process(target=worker, 
                   args=(arrayToSplit[chunksize*(nprocs-1):,:],
                         out_q, fargs), name=str(nprocs-1))
    procs.append(p)
    p.start()

    # Collect all results into a single result dict. We know how many dicts
    # with results to expect.
    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_q.get())

    # Wait for all worker processes to finish
    for p in procs:
        p.join()
    
    # order output properly
    out = resultdict[0]
    for i in range(1, nprocs):
        out = np.concatenate( (out,resultdict[i]),axis=0)
    
    #print out
    return out
