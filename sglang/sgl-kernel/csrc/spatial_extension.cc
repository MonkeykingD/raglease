/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <torch/all.h>
#include <torch/library.h>

#include "sgl_kernel_ops.h"

TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) {
  /*
   * From csrc/spatial
   */
  m.def("create_greenctx_stream_by_value(int smA, int smB, int device) -> int[]");
  m.impl("create_greenctx_stream_by_value", &create_greenctx_stream_by_value);
  m.def("create_greenctx_stream_like(int stream, int priority) -> int");
  m.impl("create_greenctx_stream_like", &create_greenctx_stream_like);
  m.def("has_greenctx_context_for_stream(int stream) -> bool");
  m.impl("has_greenctx_context_for_stream", &has_greenctx_context_for_stream);
  m.def("get_greenctx_context_for_stream(int stream) -> int");
  m.impl("get_greenctx_context_for_stream", &get_greenctx_context_for_stream);
  m.def("push_greenctx_context_for_stream(int stream) -> ()");
  m.impl("push_greenctx_context_for_stream", &push_greenctx_context_for_stream);
  m.def("pop_greenctx_context() -> ()");
  m.impl("pop_greenctx_context", &pop_greenctx_context);
}

REGISTER_EXTENSION(spatial_ops)
