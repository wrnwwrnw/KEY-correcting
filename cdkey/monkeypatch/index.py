import faiss
import time
import torch
import multiprocessing

def build(cache,index_grid,ifindex):
        # layer_count = len(cache.value_cache)
        # for i in range(layer_count):
        #    build_layer(cache.key_cache[i],i,index_grid)
        # ifindex[0] = 2
    pass
def build_layer(xb,layer_idx,index_grid):
    # with open("read.txt", "a", encoding="utf-8", buffering=1) as f:
    # f.write(f"xb: {xb}\n")
    key_np = xb.squeeze(0).detach().numpy().copy()
    key_np /= np.linalg.norm(key_np, axis=2, keepdims=True)
    for head in range(key_np.shape[0]):
        index = faiss.IndexFlatIP(key_np.shape[2])
        index.add(key_np[head])
        index_grid[layer_idx][head] = index

def search(query,layer_idx,threshold,topk,cacheidx,index_grid):
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
