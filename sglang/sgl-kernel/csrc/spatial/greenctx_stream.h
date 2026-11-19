#include <vector>
std::vector<int64_t> create_greenctx_stream_by_value(int64_t smA, int64_t smB, int64_t device);
int64_t create_greenctx_stream_like(int64_t stream, int64_t priority);
bool has_greenctx_context_for_stream(int64_t stream);
int64_t get_greenctx_context_for_stream(int64_t stream);
void push_greenctx_context_for_stream(int64_t stream);
void pop_greenctx_context();
