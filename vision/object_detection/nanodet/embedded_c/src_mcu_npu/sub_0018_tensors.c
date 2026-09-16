#include "sub_0018_tensors.h"

const TensorInfo sub_0018_tensors[] = {
  { "_split_1_command_stream", 0, 1180, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 35872, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 195200, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 195200, "FAST_SCRATCH", 0x0 },
  { "model_135_tf_compat_v1_transpose_312_transpose_70383_70780", 5, 46400, "INPUT_TENSOR", 0x0 },
  { "model_135_tf_compat_v1_transpose_311_transpose_70385_70774", 4, 46400, "INPUT_TENSOR", 0x16a80 },
  { "model_135_tf_compat_v1_transpose_315_transpose_70392", 6, 92800, "OUTPUT_TENSOR", 0x16a80 },
};

const size_t sub_0018_tensors_count = sizeof(sub_0018_tensors) / sizeof(sub_0018_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0018_address_model_135_tf_compat_v1_transpose_312_transpose_70383_70780 = 0x0;
const uint32_t sub_0018_address_model_135_tf_compat_v1_transpose_311_transpose_70385_70774 = 0x16a80;
const uint32_t sub_0018_address_model_135_tf_compat_v1_transpose_315_transpose_70392 = 0x16a80;

