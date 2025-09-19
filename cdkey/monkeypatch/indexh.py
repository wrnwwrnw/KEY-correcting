import numpy as np
import faiss
import time
import torch

def build(cache,index_grid,ifindex):

    time0 = time.time()
    layer_count = len(cache.value_cache)
    for i in range(layer_count):
        # i是层，头在嵌套内部循环
        build_layer(cache.key_cache[i],i,index_grid)
    ifindex[0] = 2
    # print("构造完成")
    # time1 = time.time()
    # print(time1-time0)
def build_layer(xb,layer_idx,index_grid):
    key_np = xb.squeeze(0).numpy()
    key_np /= np.linalg.norm(key_np, axis=2, keepdims=True)
    for head in range(key_np.shape[0]):
        index = faiss.IndexHNSWFlat(128,16, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 32
        index.add(key_np[head])
        index_grid[layer_idx][head] = index

def search(query,layer_idx,threshold,topk,cacheidx,index_grid):
    xq = query.squeeze(0).squeeze(1).to(torch.float32).cpu().numpy()
    xq /= np.linalg.norm(xq, axis=1, keepdims=True)
    for head in range(xq.shape[0]):
        if head==0:
            _ , cacheidx[layer_idx] = index_grid[layer_idx][head].search(xq[head:head+1], topk)
        else:
            _ , result = index_grid[layer_idx][head].search(xq[head:head+1], topk)
            cacheidx[layer_idx] = np.concatenate((cacheidx[layer_idx], result), axis=0)
