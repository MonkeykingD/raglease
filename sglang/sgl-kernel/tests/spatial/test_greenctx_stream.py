import pytest
import torch
from sgl_kernel.spatial import (
    create_greenctx_stream_by_value,
    create_greenctx_stream_like,
    get_greenctx_context_for_stream,
    get_sm_available,
    has_greenctx_context_for_stream,
    pop_greenctx_context,
    push_greenctx_context_for_stream,
)


def test_green_ctx():
    A = torch.randn(5120, 5120).cuda()
    B = torch.randn(5120, 5120).cuda()
    C = torch.matmul(A, B)
    sm_counts = get_sm_available(0)
    stream_group = create_greenctx_stream_by_value(sm_counts // 2, sm_counts // 2, 0)
    assert has_greenctx_context_for_stream(stream_group[0].cuda_stream)
    assert has_greenctx_context_for_stream(stream_group[1].cuda_stream)
    assert get_greenctx_context_for_stream(stream_group[0].cuda_stream) != 0
    sibling_stream = create_greenctx_stream_like(
        stream_group[0].cuda_stream, priority=-1
    )
    assert has_greenctx_context_for_stream(sibling_stream.cuda_stream)
    assert get_greenctx_context_for_stream(
        sibling_stream.cuda_stream
    ) == get_greenctx_context_for_stream(stream_group[0].cuda_stream)
    assert sibling_stream.priority <= stream_group[0].priority
    push_greenctx_context_for_stream(stream_group[0].cuda_stream)
    pop_greenctx_context()
    with torch.cuda.stream(stream_group[0]):
        for _ in range(100):
            result_0 = torch.matmul(A, B)
    with torch.cuda.stream(stream_group[1]):
        for _ in range(100):
            result_1 = torch.matmul(A, B)
    torch.cuda.synchronize()
    assert torch.allclose(result_0, C)
    assert torch.allclose(result_1, C)


if __name__ == "__main__":
    pytest.main([__file__])
