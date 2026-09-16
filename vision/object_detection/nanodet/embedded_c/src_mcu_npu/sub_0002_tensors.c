#include "sub_0002_tensors.h"

const TensorInfo sub_0002_tensors[] = {
  { "_split_1_command_stream", 0, 1184, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 11744, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 390400, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 390400, "FAST_SCRATCH", 0x0 },
  { "model_135_tf_compat_v1_transpose_247_transpose_70267_70642", 5, 92800, "INPUT_TENSOR", 0x0 },
  { "model_135_tf_compat_v1_transpose_246_transpose_70269_70636", 4, 92800, "INPUT_TENSOR", 0x2d500 },
  { "model_135_tf_compat_v1_transpose_250_transpose_70276", 6, 185600, "OUTPUT_TENSOR", 0x2d500 },
};

const size_t sub_0002_tensors_count = sizeof(sub_0002_tensors) / sizeof(sub_0002_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0002_address_model_135_tf_compat_v1_transpose_247_transpose_70267_70642 = 0x0;
const uint32_t sub_0002_address_model_135_tf_compat_v1_transpose_246_transpose_70269_70636 = 0x2d500;
const uint32_t sub_0002_address_model_135_tf_compat_v1_transpose_250_transpose_70276 = 0x2d500;

