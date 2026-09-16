#include "sub_0014_tensors.h"

const TensorInfo sub_0014_tensors[] = {
  { "_split_1_command_stream", 0, 1180, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 35920, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 195200, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 195200, "FAST_SCRATCH", 0x0 },
  { "model_135_tf_compat_v1_transpose_296_transpose_70355_70744", 5, 46400, "INPUT_TENSOR", 0x0 },
  { "model_135_tf_compat_v1_transpose_295_transpose_70357_70738", 4, 46400, "INPUT_TENSOR", 0x16a80 },
  { "model_135_tf_compat_v1_transpose_299_transpose_70364", 6, 92800, "OUTPUT_TENSOR", 0x16a80 },
};

const size_t sub_0014_tensors_count = sizeof(sub_0014_tensors) / sizeof(sub_0014_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0014_address_model_135_tf_compat_v1_transpose_296_transpose_70355_70744 = 0x0;
const uint32_t sub_0014_address_model_135_tf_compat_v1_transpose_295_transpose_70357_70738 = 0x16a80;
const uint32_t sub_0014_address_model_135_tf_compat_v1_transpose_299_transpose_70364 = 0x16a80;

