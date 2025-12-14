from memory import UnsafePointer

# ANCHOR: softmax_gpu_kernel
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from math import exp
from bit import log2_ceil
from utils.numerics import max_finite, min_finite


comptime SIZE = 128  # This must be equal to INPUT_SIZE in p18.py
comptime layout = Layout.row_major(SIZE)
comptime GRID_DIM_X = 1
# Tree-based reduction require the number of threads to be the next power of two >= SIZE for correctness.
comptime BLOCK_DIM_X = 1 << log2_ceil(SIZE)


fn softmax_gpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    # FILL IN (roughly 31 lines)
    global_i = Int(block_dim.x * block_idx.x + thread_idx.x)
    local_i = thread_idx.x

    cache_max = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_DIM_X),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    cache_sum = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_DIM_X),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    cache_exp = LayoutTensor[
        dtype,
        Layout.row_major(BLOCK_DIM_X),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()


    # fill shared memory. Note that cache_exp is pre-filled with fill(...)
    if global_i < input_size:
        # only read once from global mem...
        input_i = input[global_i]
        cache_max[local_i] = input_i
        cache_exp[local_i] = input_i
    else:
        cache_max[local_i] = 0.0
        cache_exp[local_i] = 0.0

    barrier()

    # compute max-reduction inside shared memory
    stride = UInt(BLOCK_DIM_X // 2)
    while stride > 0:
        var temp_val: output.element_type = 0.0

        # read all values first to avoid race condition
        if local_i < stride:
            temp_val = cache_max[local_i + stride]

        barrier()

        # process after read
        if local_i < stride:
            cache_max[local_i] = max(cache_max[local_i], temp_val)

        barrier()

        stride //= 2

    max_x = cache_max[0]
    # using the computed max values, compute exp(x_i - max(x))
    # fill the caches with that value.
    if global_i < input_size:
        x_i_exp = exp(cache_exp[local_i] - max_x)
        cache_exp[local_i] = x_i_exp
        cache_sum[local_i] = x_i_exp
    barrier()

    # compute sum-reduction inside shared memory
    stride = UInt(BLOCK_DIM_X // 2)
    while stride > 0:
        var temp_val: output.element_type = 0.0

        # read all values first to avoid race condition
        if local_i < stride:
            temp_val = cache_sum[local_i + stride]

        barrier()

        # process after read
        if local_i < stride:
            cache_sum[local_i] += temp_val

        barrier()

        stride //= 2

    sum_x = cache_sum[0]
    # compute softmax term and write output
    if global_i < input_size:
        output[global_i] = cache_exp[local_i] / sum_x






# ANCHOR_END: softmax_gpu_kernel


# ANCHOR: softmax_cpu_kernel
fn softmax_cpu_kernel[
    layout: Layout,
    input_size: Int,
    dtype: DType = DType.float32,
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    input: LayoutTensor[dtype, layout, ImmutAnyOrigin],
):
    # FILL IN (roughly 10 lines)

    var max_x: input.element_type = 0
    for i in range(input_size):
        if input[i] > max_x:
            max_x = input[i]


    var sum_all: input.element_type = 0
    for i in range(input_size):
        sum_all += exp(input[i] - max_x)


    for i in range(input_size):
        output[i] = exp(input[i] - max_x) / sum_all


# ANCHOR_END: softmax_cpu_kernel

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor


@compiler.register("softmax")
struct SoftmaxCustomOp:
    @staticmethod
    fn execute[
        target: StaticString,  # "cpu" or "gpu"
        input_size: Int,
        dtype: DType = DType.float32,
    ](
        output: OutputTensor[rank=1],
        input: InputTensor[rank = output.rank],
        ctx: DeviceContextPtr,
    ) raises:
        # Note: rebind is necessary now but it shouldn't be!
        var output_tensor = rebind[LayoutTensor[dtype, layout, MutAnyOrigin]](
            output.to_layout_tensor()
        )
        var input_tensor = rebind[LayoutTensor[dtype, layout, ImmutAnyOrigin]](
            input.to_layout_tensor()
        )

        @parameter
        if target == "gpu":
            gpu_ctx = ctx.get_device_context()
            # making sure the output tensor is zeroed out before the kernel is called
            gpu_ctx.enqueue_memset(
                DeviceBuffer[output_tensor.dtype](
                    gpu_ctx,
                    rebind[LegacyUnsafePointer[Scalar[output_tensor.dtype]]](
                        output_tensor.ptr
                    ),
                    input_size,
                    owning=False,
                ),
                0,
            )

            comptime kernel = softmax_gpu_kernel[layout, input_size, dtype]
            gpu_ctx.enqueue_function_checked[kernel, kernel](
                output_tensor,
                input_tensor,
                grid_dim=GRID_DIM_X,
                block_dim=BLOCK_DIM_X,
            )

        elif target == "cpu":
            softmax_cpu_kernel[layout, input_size, dtype](
                output_tensor, input_tensor
            )
        else:
            raise Error("Unsupported target: " + target)
