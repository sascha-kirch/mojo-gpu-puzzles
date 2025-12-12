from gpu import thread_idx, block_idx, block_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from sys import size_of, argv
from math import log2
from testing import assert_equal

# ANCHOR: prefix_sum_simple
comptime TPB = 8
comptime SIZE = 8
comptime BLOCKS_PER_GRID = (1, 1)
comptime THREADS_PER_BLOCK = (TPB, 1)
comptime dtype = DType.float32
comptime layout = Layout.row_major(SIZE)


fn prefix_sum_simple[
    layout: Layout
](
    output: LayoutTensor[dtype, layout, MutAnyOrigin],
    a: LayoutTensor[dtype, layout, ImmutAnyOrigin],
    size: UInt,
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    # FILL ME IN (roughly 18 lines)
    shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    if global_i < size:
        shared[local_i] = a[global_i]

    barrier()

    # My initial solution:
    # ---------------------------------------------------------------------------
    # Issues:
    # - no read, sync, write pattern might cause race conditions!!!
    # - no check that the local_i is < size
    # - i use a while loop that checks conditions instead of a for loop that considers the log
    # if global_i < size:
    #     var step = 1

    #     while step <= Int(size/2):
    #         # update share memory
    #         if local_i > step:
    #             shared[local_i] += shared[local_i - step]

    #         barrier()
    #         step *= 2

    # ---------------------------------------------------------------------------
    # Version considering solution:
    var step = UInt(1)
    for i in range(Int(log2(Scalar[dtype](TPB)))):
        var current_val: output.element_type = 0

        if local_i >= step and local_i < size:
            current_val = shared[local_i - step] # read
        barrier()

        if local_i >= step and local_i < size:
            shared[local_i] += current_val # write
        barrier()

        step *= 2

    # ---------------------------------------------------------------------------
    # write processed shared memory back to output tensor
    output[global_i] = shared[local_i]


# ANCHOR_END: prefix_sum_simple

# ANCHOR: prefix_sum_complete
comptime SIZE_2 = 15
comptime BLOCKS_PER_GRID_2 = (2, 1)
comptime THREADS_PER_BLOCK_2 = (TPB, 1)
comptime EXTENDED_SIZE = SIZE_2 + 2  # up to 2 blocks
comptime layout_2 = Layout.row_major(SIZE_2)
comptime extended_layout = Layout.row_major(EXTENDED_SIZE)


# Kernel 1: Compute local prefix sums and store block sums in out
fn prefix_sum_local_phase[
    out_layout: Layout, in_layout: Layout
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, in_layout, ImmutAnyOrigin],
    size: UInt,
):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    local_i = thread_idx.x
    # FILL ME IN (roughly 20 lines)
    shared = LayoutTensor[
        dtype,
        Layout.row_major(TPB),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    if global_i < size:
        shared[local_i] = a[global_i]

    barrier()

    var step = UInt(1)

    @parameter
    for i in range(Int(log2(Scalar[dtype](TPB)))):
        var current_val: output.element_type = 0

        if local_i >= step and local_i < TPB:
            current_val = shared[local_i - step] # read
        barrier()

        if local_i >= step and local_i < TPB:
            shared[local_i] += current_val # write
        barrier()

        step *= 2

    # write data block
    if global_i < size:
        output[global_i] = shared[local_i]

    # write block sum into extended buffer
    if local_i == TPB - 1:
        idx_block_sum = size + block_idx.x
        output[idx_block_sum] = shared[local_i]


# Kernel 2: Add block sums to their respective blocks
fn prefix_sum_block_sum_phase[
    layout: Layout
](output: LayoutTensor[dtype, layout, MutAnyOrigin], size: UInt):
    global_i = block_dim.x * block_idx.x + thread_idx.x
    # FILL ME IN (roughly 3 lines)
    if global_i < size and block_idx.x > 0:
        idx_prev_block_sum = size + block_idx.x - 1
        output[global_i] += output[idx_prev_block_sum]

# ANCHOR_END: prefix_sum_complete


def main():
    with DeviceContext() as ctx:
        if len(argv()) != 2 or argv()[1] not in [
            "--simple",
            "--complete",
        ]:
            raise Error(
                "Expected one command-line argument: '--simple' or '--complete'"
            )

        use_simple = argv()[1] == "--simple"

        size = SIZE if use_simple else SIZE_2
        num_blocks = (size + TPB - 1) // TPB

        if not use_simple and num_blocks > EXTENDED_SIZE - SIZE_2:
            raise Error("Extended buffer too small for the number of blocks")

        buffer_size = size if use_simple else EXTENDED_SIZE
        out = ctx.enqueue_create_buffer[dtype](buffer_size)
        out.enqueue_fill(0)
        a = ctx.enqueue_create_buffer[dtype](size)
        a.enqueue_fill(0)

        with a.map_to_host() as a_host:
            for i in range(size):
                a_host[i] = i

        if use_simple:
            a_tensor = LayoutTensor[dtype, layout, ImmutAnyOrigin](a)
            out_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](out)

            comptime kernel = prefix_sum_simple[layout]
            ctx.enqueue_function_checked[kernel, kernel](
                out_tensor,
                a_tensor,
                UInt(size),
                grid_dim=BLOCKS_PER_GRID,
                block_dim=THREADS_PER_BLOCK,
            )
        else:
            var a_tensor = LayoutTensor[dtype, layout_2, ImmutAnyOrigin](a)
            var out_tensor = LayoutTensor[dtype, extended_layout, MutAnyOrigin](
                out
            )

            # ANCHOR: prefix_sum_complete_block_level_sync
            # Phase 1: Local prefix sums
            comptime kernel = prefix_sum_local_phase[extended_layout, layout_2]
            ctx.enqueue_function_checked[kernel, kernel](
                out_tensor,
                a_tensor,
                UInt(size),
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )

            # Wait for all `blocks` to complete with using host `ctx.synchronize()`
            # Note this is in contrast with using `barrier()` in the kernel
            # which is a synchronization point for all threads in the same block and not across blocks.
            ctx.synchronize()

            # Phase 2: Add block sums
            comptime kernel2 = prefix_sum_block_sum_phase[extended_layout]
            ctx.enqueue_function_checked[kernel2, kernel2](
                out_tensor,
                UInt(size),
                grid_dim=BLOCKS_PER_GRID_2,
                block_dim=THREADS_PER_BLOCK_2,
            )
            # ANCHOR_END: prefix_sum_complete_block_level_sync

        # Verify results for both cases
        expected = ctx.enqueue_create_host_buffer[dtype](size)
        expected.enqueue_fill(0)
        ctx.synchronize()

        with a.map_to_host() as a_host:
            expected[0] = a_host[0]
            for i in range(1, size):
                expected[i] = expected[i - 1] + a_host[i]

        with out.map_to_host() as out_host:
            if not use_simple:
                print(
                    "Note: we print the extended buffer here, but we only need"
                    " to print the first `size` elements"
                )

            print("out:", out_host)
            print("expected:", expected)
            # Here we need to use the size of the original array, not the extended one
            size = size if use_simple else SIZE_2
            for i in range(size):
                assert_equal(out_host[i], expected[i])
