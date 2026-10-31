// Growable device buffers (CUDA virtual memory management): a buffer reserves address space for its
// full size and maps physical memory into it on demand, so a sequence's attention caches use memory
// for the tokens they hold, while CUDA graphs keep the fixed addresses. Results are CUresult codes.

#include <cuda.h>
#include <cuda_runtime.h>

namespace {

CUmemAllocationProp device_prop(int device) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    return prop;
}

int current_device() {
    int device = 0;
    cudaGetDevice(&device);
    return device;
}

}  // namespace

extern "C" int quench6k_vm_granularity(size_t* out) {
    CUmemAllocationProp prop = device_prop(current_device());
    return cuMemGetAllocationGranularity(out, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
}

extern "C" int quench6k_vm_reserve(size_t bytes, unsigned long long* addr) {
    CUdeviceptr p = 0;
    CUresult r = cuMemAddressReserve(&p, bytes, 0, 0, 0);
    *addr = p;
    return r;
}

extern "C" int quench6k_vm_free(unsigned long long addr, size_t bytes) { return cuMemAddressFree(addr, bytes); }

// Backs [addr, addr + bytes) with new physical memory; *handle receives it for quench6k_vm_unmap.
extern "C" int quench6k_vm_map(unsigned long long addr, size_t bytes, unsigned long long* handle) {
    int device = current_device();
    CUmemAllocationProp prop = device_prop(device);
    CUmemGenericAllocationHandle h = 0;
    CUresult r = cuMemCreate(&h, bytes, &prop, 0);
    if (r != CUDA_SUCCESS) return r;
    r = cuMemMap(addr, bytes, 0, h, 0);
    if (r != CUDA_SUCCESS) {
        cuMemRelease(h);
        return r;
    }
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    r = cuMemSetAccess(addr, bytes, &access, 1);
    if (r != CUDA_SUCCESS) {
        cuMemUnmap(addr, bytes);
        cuMemRelease(h);
        return r;
    }
    *handle = h;
    return CUDA_SUCCESS;
}

extern "C" int quench6k_vm_unmap(unsigned long long addr, size_t bytes, unsigned long long handle) {
    CUresult r = cuMemUnmap(addr, bytes);
    CUresult r2 = cuMemRelease(handle);
    return r != CUDA_SUCCESS ? r : r2;
}
