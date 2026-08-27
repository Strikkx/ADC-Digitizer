# ADC-Digitizer

GPU-accelerated correlation processing for high-speed digitizer data, built with CUDA C++. Developed for a quantum spectroscopy data pipeline where raw ADC samples need to be processed fast enough to keep up with acquisition — the target is on the order of **10 billion multiplications per second**.

## Background

Digitizer hardware streams samples as a single interleaved 1D array in the form:

```
a1, b1, a2, b2, a3, b3, a4, b4, ...
```

Each pair `(a, b)` corresponds to two correlated channels. The processing pipeline performs three steps per adjacent pair-of-pairs:

1. **Subtraction** — `a1 - a2`, `a3 - a4`, ... and `b1 - b2`, `b3 - b4`, ...
2. **Multiplication** — `(a1 - a2) * (b1 - b2)`, `(a3 - a4) * (b3 - b4)`, ...
3. **Summation** — sum of all the products above into a single scalar

This is a standard building block in correlation-based quantum spectroscopy measurements, where the raw signal needs this transform applied in real time before it's usable.

## Repository Structure

```
ADC-Digitizer/
├── ADC Digitizer.sln       # Visual Studio solution
└── ADC Digitizer/
    ├── kernel.cu           # Current implementation
    └── kernelold.cu        # Earlier prototypes / experiments (kept for reference)
```

- **`kernel.cu`** — the active implementation. Runs subtraction, multiplication, and summation as three separate kernels, each using shared memory and a tree-based block-level reduction, and times each stage independently with CUDA events.
- **`kernelold.cu`** — earlier iterations of the same pipeline, including a version with warp-level `__shfl_down_sync` reduction and separate fused sum kernels. Left in the repo as a record of prior approaches and performance comparisons; not the recommended entry point.

## Requirements

- NVIDIA GPU with CUDA support (developed/tested against Ampere-architecture cards, e.g. RTX A4000)
- CUDA Toolkit (matching your driver version)
- Visual Studio with CUDA project support (the repo ships as a `.sln`)
- [Nsight Systems](https://developer.nvidia.com/nsight-systems) / Nsight Compute, for profiling and identifying the limiting kernel/step

## Building

1. Open `ADC Digitizer.sln` in Visual Studio.
2. Ensure the CUDA Toolkit version referenced by the project matches what's installed locally.
3. Build in Release mode for meaningful timing numbers — Debug builds will not reflect real throughput.

## Running

The current `main()` in `kernel.cu`:

1. Generates a random test array of `N = 1,000,000` floats (standing in for real digitizer output).
2. Copies the array to device memory.
3. Launches the subtraction, multiplication, and summation kernels, timing each individually with `cudaEvent_t` start/stop pairs.
4. Copies partial per-block results back and reduces them on the host into a final scalar.
5. Prints per-kernel timing plus total time and the final result.

Adjust `n` and `BLOCK_SIZE` in `kernel.cu` to match your actual data size and target GPU.

## Performance Notes

- Reduction is implemented with shared memory and a binary-tree reduction pattern (halving active threads each step) inside each kernel.
- `BLOCK_SIZE` is currently fixed at compile time — tune this per-GPU architecture.
- Use Nsight Compute to check achieved occupancy, memory throughput, and warp execution efficiency before assuming a kernel is compute-bound; on Ampere-class GPUs, an underperforming kernel is often left-on-the-table warp/memory efficiency rather than a hardware ceiling.
- Next optimization steps to explore: fusing subtraction + multiplication + summation into a single kernel to cut global memory round-trips, warp-shuffle-based reduction instead of shared-memory tree reduction (see `kernelold.cu` for a prior attempt), and loop unrolling in the per-thread accumulation loop.

## Status

Actively being optimized as part of ongoing GPU-acceleration work for a quantum spectroscopy data pipeline. Benchmarks and Nsight reports are used to guide iteration between `kernel.cu` and comparisons against `kernelold.cu`.
