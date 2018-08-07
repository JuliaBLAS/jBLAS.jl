


# const architecture = CpuId.cpuarchitecture()#Really necessary?
const REGISTER_SIZE = CpuId.simdbytes()
# const REGISTER_COUNT = CpuId.cpufeature(CpuId.AVX512F) ? 32 : 16
# const AVX512FMA = CpuId.cpufeature(CpuId.AVX512IFMA) #
const REGISTER_COUNT = CpuId.cpufeature(CpuId.AVX512IFMA) ? 32 : 16
const FP256 = CpuId.cpufeature(CpuId.FP256) # Is AVX2 fast?
const CACHELINE_SIZE = CpuId.cachelinesize()
const CACHE_SIZE = CpuId.cachesize()
const NUM_CORES = CpuId.cpucores()
### So, the goal here moving forward will be to take advantage of Julia's flexibility / 
### JIT compiling for the specific machine it's running on, to create algorithms
### flexibly optimized for the architecture's properties that they are running on.
### 
### for a cache size, need to pick how many blocks of "D" we want to fit
### and also what fraction of "N". This sounds like a discrete optimization
### problem. 
###
