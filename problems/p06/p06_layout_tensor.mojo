from memory import UnsafePointer
from gpu import thread_idx, block_idx, block_dim
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from testing import assert_equal

# ANCHOR: add_10_blocks
comptime SIZE = 9
comptime BLOCKS_PER_GRID = (3, 1)
comptime THREADS_PER_BLOCK = (4, 1)
comptime out_layout = Layout.row_major(SIZE)
comptime a_layout = Layout.row_major(SIZE)
comptime dtype = DType.float32


fn add_10_blocks[
    out_layout: Layout,
    a_layout: Layout,
](
    output: LayoutTensor[dtype, out_layout, MutAnyOrigin],
    a: LayoutTensor[dtype, a_layout, MutAnyOrigin],
    size: UInt,
):
    i = block_dim.x * block_idx.x + thread_idx.x
    # FILL ME IN (roughly 2 lines)
    if i < size:
        output[i] = a[i] + 10


# ANCHOR_END: add_10_blocks


def main():
    with DeviceContext() as ctx:
        out_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        out_buf.enqueue_fill(0)
        out_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](out_buf)
        a_buf = ctx.enqueue_create_buffer[dtype](SIZE)
        a_buf.enqueue_fill(0)
        with a_buf.map_to_host() as a_host:
            for i in range(SIZE):
                a_host[i] = i
        a_tensor = LayoutTensor[dtype, out_layout, MutAnyOrigin](a_buf)

        comptime kernel = add_10_blocks[out_layout, a_layout]

        ctx.enqueue_function_checked[kernel, kernel](
            out_tensor,
            a_tensor,
            UInt(SIZE),
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        expected = ctx.enqueue_create_host_buffer[dtype](SIZE)
        expected.enqueue_fill(0)

        ctx.synchronize()

        for i in range(SIZE):
            expected[i] = i + 10

        with out_buf.map_to_host() as out_buf_host:
            print("out:", out_buf_host)
            print("expected:", expected)
            for i in range(SIZE):
                assert_equal(out_buf_host[i], expected[i])
