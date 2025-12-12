from memory import UnsafePointer, stack_allocation
from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from sys import size_of
from testing import assert_equal

# ANCHOR: dot_product
comptime TPB = 8
comptime SIZE = 8
comptime BLOCKS_PER_GRID = (1, 1)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime dtype = DType.float32


fn dot_product(
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    size: UInt,
):
    # FILL ME IN (roughly 13 lines)
    shared = stack_allocation[
        TPB,
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()

    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x

    if global_i < size:
        shared[local_i] = a[global_i] * b[global_i]

    barrier()

    c:Scalar[dtype] = 0.0

    # Naive approach: only 1 thread performs the reduction....
    # -------------------------------------------------------
    # if global_i == 0:
    #     for l in range(size):
    #         c += shared[l]
    #     output[0] = c

    # Better approach: parallel reduction in shared memory. single thread writes the output
    # Initial:  [0*0  1*1  2*2  3*3  4*4  5*5  6*6  7*7]
    #         = [0    1    4    9    16   25   36   49]

    # Step 1:   [0+16 1+25 4+36 9+49  16   25   36   49] involved threads: 4, stride: 4
    #         = [16   26   40   58   16   25   36   49]
    #            ^    ^    ^    ^    ^    ^    ^    ^
    #            |    |    |    |    |    |    |    |
    #            t0   t1   t2   t3   t0   t1   t2   t3
    #           barrier() => 4 processes wait until other 4 are done!

    # Step 2:   [16+40 26+58 40   58   16   25   36   49] involved threads: 2, stride: 2
    #         = [56   84   40   58   16   25   36   49]
    #            ^    ^    ^    ^
    #            |    |    |    |
    #            t0   t1   t0   t1
    #           barrier()

    # Step 3:   [56+84  84   40   58   16   25   36   49] involved threads: 1, stride: 1
    #         = [140   84   40   58   16   25   36   49]
    #            ^     ^
    #            |     |
    #            t0    t0
    #           barrier()

    # stride => 0, break out of while
    # -------------------------------------------------------

    stride = UInt(TPB // 2)
    while stride < 0:
        if local_i < stride:
            shared[local_i] += shared[local_i + stride]

        barrier()

        stride //= 2

    if local_i == 0:
        output[0] = shared[0]

# ANCHOR_END: dot_product


def main():
    with DeviceContext() as ctx:
        out = ctx.enqueue_create_buffer[dtype](1)
        out.enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](SIZE)
        a.enqueue_fill(0)
        b = ctx.enqueue_create_buffer[dtype](SIZE)
        b.enqueue_fill(0)
        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                a_host[i] = i
                b_host[i] = i

        ctx.enqueue_function_checked[dot_product, dot_product](
            out,
            a,
            b,
            UInt(SIZE),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        expected = ctx.enqueue_create_host_buffer[dtype](1)
        expected.enqueue_fill(0)

        ctx.synchronize()

        with a.map_to_host() as a_host, b.map_to_host() as b_host:
            for i in range(SIZE):
                expected[0] += a_host[i] * b_host[i]

        with out.map_to_host() as out_host:
            print("out:", out_host)
            print("expected:", expected)
            assert_equal(out_host[0], expected[0])
