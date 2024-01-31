import os
import struct
import time

import numpy as np
from diskannpy import *

from ..base.module import BaseANN


class DiskANN(BaseANN):
    def __init__(self, metric, param):
        self.metric = {"angular": "cosine", "euclidean": "l2"}[metric]
        self.l_build = int(param["l_build"])
        self.max_outdegree = int(param["max_outdegree"])
        self.alphas = [float(alpha) for alpha in param["alphas"]] 
        print("DiskANN: L_Build = " + str(self.l_build))
        print("DiskANN: R = " + str(self.max_outdegree))
        print("DiskANN: Alpha = {}".format(self.alphas[0]))

        self.num_threads = 8
        
        '''
        self.params = vp.Parameters()
        self.params.set("L", self.l_build)
        self.params.set("R", self.max_outdegree)
        self.params.set("C", 750)
        self.params.set("alpha", self.alpha)
        self.params.set("saturate_graph", False)
        self.params.set("num_threads", 1)
        '''

    def build(
        self,
        metric,
        dtype_str,
        alpha,
        index_directory,
        indexdata_file,
        Lb,
        graph_degree,
        num_threads,
        index_prefix,
    ):
        """
        :param metric:
        :param dtype_str:
        :param index_directory:
        :param indexdata_file:
        :param querydata_file:
        :param Lb:
        :param graph_degree:
        :param K:
        :param Ls:
        :param num_threads:
        :param gt_file:
        :param index_prefix:
        :param search_only:
        """
        if dtype_str == "float":
            dtype = np.single
        elif dtype_str == "int8":
            dtype = np.byte
        elif dtype_str == "uint8":
            dtype = np.ubyte
        else:
            raise ValueError("data_type must be float, int8 or uint8")

        # build index
  
        build_memory_index(
            data=indexdata_file,
            distance_metric=metric,
            vector_dtype=dtype,
            index_directory=index_directory,
            complexity=Lb,
            graph_degree=graph_degree,
            num_threads=num_threads,
            index_prefix=index_prefix,
            alpha=alpha,
            use_pq_build=False,
            num_pq_bytes=8,
            use_opq=False,
        )
          

        index = StaticMemoryIndex(
            distance_metric=metric,
            vector_dtype=dtype,
            index_directory=index_directory,
            num_threads=num_threads,  # this can be different at search time if you would like
            initial_search_complexity=Lb,
            index_prefix=index_prefix
        )


        return index


    def fit(self, X):
        def bin_to_float(binary):
            return struct.unpack("!f", struct.pack("!I", int(binary, 2)))[0]

        print("DiskANN: Starting Fit...")
        index_dir = "indices"

        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

        data_path = os.path.join(index_dir, "base.bin")
        #self.name = "Vamana-{}-{}-[{},{},{},{}]".format(self.l_build, self.max_outdegree, self.alphas[0],self.alphas[1],self.alphas[2],self.alphas[3])
        self.name = "DiskANN-{}-{}-{}".format(self.l_build, self.max_outdegree, self.alphas[0])

        save_path = os.path.join(index_dir, self.name)
        print("DiskANN: Index Stored At: " + save_path)
        shape = [
            np.float32(bin_to_float("{:032b}".format(X.shape[0]))),
            np.float32(bin_to_float("{:032b}".format(X.shape[1]))),
        ]
        X = X.flatten()
        X = np.insert(X, 0, shape)
        X.tofile(data_path)

        if not os.path.exists(save_path):
            print("DiskANN: Creating Index")
            s = time.time()
            if self.metric == "l2":
                #index = vp.SinglePrecisionIndex(vp.Metric.FAST_L2, data_path)
                self.index = self.build(self.metric, "float", self.alphas[0], index_dir, data_path,self.l_build, self.max_outdegree, self.num_threads, self.name)
            elif self.metric == "cosine":
                self.index = self.build(self.metric, "float", self.alphas[0], index_dir, data_path,self.l_build, self.max_outdegree, self.num_threads, self.name)
          
            else:
                print("DiskANN: Unknown Metric Error!")

            
            #index.build(self.params, [])
            t = time.time()
            print("DiskANN: Index Build Time (sec) = " + str(t - s))
            #index.save(save_path)

        if os.path.exists(save_path):
            print("DiskANN: Loading Index: " + str(save_path))
            s = time.time()
            if self.metric == "l2":
                self.index = StaticMemoryIndex(distance_metric=self.metric,vector_dtype="float",index_directory=index_dir, num_threads=self.num_threads, initial_search_complexity=self.l_build,index_prefix=self.name)
            elif self.metric == "cosine":
                self.index = StaticMemoryIndex(distance_metric=self.metric,vector_dtype="float",index_directory=index_dir, num_threads=self.num_threads, initial_search_complexity=self.l_build,index_prefix=self.name)
            else:
                print("DiskANN: Unknown Metric Error!")
            #self.index.load(file_name=save_path)
            print("DiskANN: Index Loaded")
            #self.index.optimize_graph()
            #print("Vamana: Graph Optimization Completed")
            t = time.time()
            print("DiskANN: Index Load Time (sec) = " + str(t - s))
        else:
            print("DiskANN: Unexpected Index Build Time Error")

        print("DiskANN: End of Fit")

    def set_query_arguments(self, l_search):
        print("DiskANN: L_Search = " + str(l_search))
        self.l_search = l_search

    def query(self, v, n):
        return self.index.search(v, n, self.l_search).identifiers

    def batch_query(self, X, n):
        self.num_queries = X.shape[0]
        self.result = self.index.batch_search(X, n,  self.l_search, self.num_threads).identifiers

    def get_batch_results(self):
        return self.result
