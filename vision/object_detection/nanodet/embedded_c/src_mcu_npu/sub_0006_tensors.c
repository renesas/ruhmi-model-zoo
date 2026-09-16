#include "sub_0006_tensors.h"

const TensorInfo sub_0006_tensors[] = {
  { "_split_1_command_stream", 0, 1184, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 11872, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 390400, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 390400, "FAST_SCRATCH", 0x0 },
  { "model_135_tf_compat_v1_transpose_263_transpose_70295_70678", 5, 92800, "INPUT_TENSOR", 0x0 },
  { "model_135_tf_compat_v1_transpose_262_transpose_70297_70672", 4, 92800, "INPUT_TENSOR", 0x2d500 },
  { "model_135_tf_compat_v1_transpose_266_transpose_70304", 6, 185600, "OUTPUT_TENSOR", 0x2d500 },
};

const size_t sub_0006_tensors_count = sizeof(sub_0006_tensors) / sizeof(sub_0006_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0006_address_model_135_tf_compat_v1_transpose_263_transpose_70295_70678 = 0x0;
const uint32_t sub_0006_address_model_135_tf_compat_v1_transpose_262_transpose_70297_70672 = 0x2d500;
const uint32_t sub_0006_address_model_135_tf_compat_v1_transpose_266_transpose_70304 = 0x2d500;

