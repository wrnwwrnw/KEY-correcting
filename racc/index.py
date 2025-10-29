import numpy as np
import faiss
import time
import torch
import multiprocessing

#flat
def flat_build(cache,index_grid,ifindex):
        layer_count = 32
        for i in range(layer_count):
            # i是层，头在嵌套内部循环
            flat_build_layer(cache.layers[i].keys,i,index_grid)
        ifindex[0] = 2
    # pass
    # print("构造完成")
    # time1 = time.time()
    # print(time1-time0)
def flat_build_layer(xb,layer_idx,index_grid):
    key_np = xb.squeeze(0).detach().numpy().copy()
    key_np /= np.linalg.norm(key_np, axis=2, keepdims=True)
    for head in range(key_np.shape[0]):
        index = faiss.IndexFlatIP(key_np.shape[2])
        index.add(key_np[head])
        index_grid[layer_idx][head] = index

def flat_search(query,layer_idx,threshold,topk,cacheidx,index_grid,cache):
    topk = min(topk,cache.layers[layer_idx].keys.shape[2])
    xq = query.squeeze(0).squeeze(1).to(torch.float32).cpu().numpy()
    xq /= np.linalg.norm(xq, axis=1, keepdims=True)
    for head in range(xq.shape[0]):
        if head==0:
            _ , cacheidx[layer_idx] = index_grid[layer_idx][head].search(xq[head:head+1], topk)
            cacheidx[layer_idx] = cacheidx[layer_idx].astype(np.int64)
        else:
            _ , result = index_grid[layer_idx][head].search(xq[head:head+1], topk)
            result = result.astype(np.int64)
            cacheidx[layer_idx] = np.concatenate((cacheidx[layer_idx], result), axis=0)

# IVF-PQ
def ivfpq_build(cache, index_grid, ifindex):
    layer_count = 32
    for i in range(layer_count):
        ivfpq_build_layer(cache.layers[i].keys, i, index_grid)
    ifindex[0] = 2

def ivfpq_build_layer(xb, layer_idx, index_grid):
    key_np = xb.squeeze(0).detach().numpy().copy()
    key_np /= np.linalg.norm(key_np, axis=2, keepdims=True)
    
    # 固定参数
    nlist = 100  # 聚类中心数量
    m = 8        # PQ子向量数量
    nbits = 8    # 每个子向量的比特数
    
    for head in range(key_np.shape[0]):
        d = key_np.shape[2]  # 向量维度
        
        # 创建IVF-PQ索引
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
        
        # 如果数据量小于nlist，调整nlist
        n_vectors = key_np[head].shape[0]
        actual_nlist = min(nlist, n_vectors // 2) if n_vectors >= 2 else 1
        
        if actual_nlist != nlist:
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFPQ(quantizer, d, actual_nlist, m, nbits)
        
        # 训练并添加向量
        if not index.is_trained:
            index.train(key_np[head])
        index.add(key_np[head])
        
        index_grid[layer_idx][head] = index

def ivfpq_search(query, layer_idx, threshold, topk, cacheidx, index_grid, cache):
    topk = min(topk, cache.layers[layer_idx].keys.shape[2])
    xq = query.squeeze(0).squeeze(1).to(torch.float32).cpu().numpy()
    xq /= np.linalg.norm(xq, axis=1, keepdims=True)
    
    nprobe = 10  # 固定搜索的聚类中心数量
    
    for head in range(xq.shape[0]):
        # 设置nprobe参数
        index_grid[layer_idx][head].nprobe = nprobe
        
        if head == 0:
            _, cacheidx[layer_idx] = index_grid[layer_idx][head].search(xq[head:head+1], topk)
            cacheidx[layer_idx] = cacheidx[layer_idx].astype(np.int64)
        else:
            _, result = index_grid[layer_idx][head].search(xq[head:head+1], topk)
            result = result.astype(np.int64)
            cacheidx[layer_idx] = np.concatenate((cacheidx[layer_idx], result), axis=0)


# HNSW 实现 
def hnsw_build(cache, index_grid, ifindex):
    layer_count = 32
    for i in range(layer_count):
        hnsw_build_layer(cache.layers[i].keys, i, index_grid)
    ifindex[0] = 2

def hnsw_build_layer(xb, layer_idx, index_grid):
    key_np = xb.squeeze(0).detach().numpy().copy()
    key_np /= np.linalg.norm(key_np, axis=2, keepdims=True)
    
    # 固定参数
    M = 32              # HNSW连接数
    efConstruction = 200  # 构建时搜索深度
    
    for head in range(key_np.shape[0]):
        d = key_np.shape[2]  # 向量维度
        
        # 创建HNSW索引
        index = faiss.IndexHNSWFlat(d, M)
        index.hnsw.efConstruction = efConstruction
        
        # HNSW不需要训练，直接添加
        index.add(key_np[head])
        
        index_grid[layer_idx][head] = index

def hnsw_search(query, layer_idx, threshold, topk, cacheidx, index_grid, cache):
    topk = min(topk, cache.layers[layer_idx].keys.shape[2])
    xq = query.squeeze(0).squeeze(1).to(torch.float32).cpu().numpy()
    xq /= np.linalg.norm(xq, axis=1, keepdims=True)
    
    efSearch = 100  # 固定搜索时的搜索深度
    
    for head in range(xq.shape[0]):
        # 设置efSearch参数
        index_grid[layer_idx][head].hnsw.efSearch = efSearch
        
        if head == 0:
            _, cacheidx[layer_idx] = index_grid[layer_idx][head].search(xq[head:head+1], topk)
            cacheidx[layer_idx] = cacheidx[layer_idx].astype(np.int64)
        else:
            _, result = index_grid[layer_idx][head].search(xq[head:head+1], topk)
            result = result.astype(np.int64)
            cacheidx[layer_idx] = np.concatenate((cacheidx[layer_idx], result), axis=0)
