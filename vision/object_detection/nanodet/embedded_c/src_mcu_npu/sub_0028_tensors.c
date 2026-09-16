#include "sub_0028_tensors.h"

const TensorInfo sub_0028_tensors[] = {
  { "_split_1_command_stream", 0, 1188, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 122544, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 94400, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 94400, "FAST_SCRATCH", 0x0 },
  { "model_135_tf_compat_v1_transpose_353_transpose_70457_70864", 5, 23200, "INPUT_TENSOR", 0x0 },
  { "model_135_tf_compat_v1_transpose_352_transpose_70459_70858", 4, 23200, "INPUT_TENSOR", 0xb540 },
  { "model_135_tf_compat_v1_transpose_356_transpose_70466", 6, 46400, "OUTPUT_TENSOR", 0xb540 },
};

const size_t sub_0028_tensors_count = sizeof(sub_0028_tensors) / sizeof(sub_0028_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0028_address_model_135_tf_compat_v1_transpose_353_transpose_70457_70864 = 0x0;
const uint32_t sub_0028_address_model_135_tf_compat_v1_transpose_352_transpose_70459_70858 = 0xb540;
const uint32_t sub_0028_address_model_135_tf_compat_v1_transpose_356_transpose_70466 = 0xb540;

