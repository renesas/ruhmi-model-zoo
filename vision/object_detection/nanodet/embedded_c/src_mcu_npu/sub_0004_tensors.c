#include "sub_0004_tensors.h"

const TensorInfo sub_0004_tensors[] = {
  { "_split_1_command_stream", 0, 1180, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 11712, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 390400, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 390400, "FAST_SCRATCH", 0x0 },
  { "model_135_tf_compat_v1_transpose_255_transpose_70281_70660", 5, 92800, "INPUT_TENSOR", 0x0 },
  { "model_135_tf_compat_v1_transpose_254_transpose_70283_70654", 4, 92800, "INPUT_TENSOR", 0x2d500 },
  { "model_135_tf_compat_v1_transpose_258_transpose_70290", 6, 185600, "OUTPUT_TENSOR", 0x2d500 },
};

const size_t sub_0004_tensors_count = sizeof(sub_0004_tensors) / sizeof(sub_0004_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0004_address_model_135_tf_compat_v1_transpose_255_transpose_70281_70660 = 0x0;
const uint32_t sub_0004_address_model_135_tf_compat_v1_transpose_254_transpose_70283_70654 = 0x2d500;
const uint32_t sub_0004_address_model_135_tf_compat_v1_transpose_258_transpose_70290 = 0x2d500;

