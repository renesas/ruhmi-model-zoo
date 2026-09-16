#include "sub_0012_tensors.h"

const TensorInfo sub_0012_tensors[] = {
  { "_split_1_command_stream", 0, 1176, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 35488, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 195200, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 195200, "FAST_SCRATCH", 0x0 },
  { "model_135_tf_compat_v1_transpose_288_transpose_70341_70726", 5, 46400, "INPUT_TENSOR", 0x0 },
  { "model_135_tf_compat_v1_transpose_287_transpose_70343_70720", 4, 46400, "INPUT_TENSOR", 0x16a80 },
  { "model_135_tf_compat_v1_transpose_291_transpose_70350", 6, 92800, "OUTPUT_TENSOR", 0x16a80 },
};

const size_t sub_0012_tensors_count = sizeof(sub_0012_tensors) / sizeof(sub_0012_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0012_address_model_135_tf_compat_v1_transpose_288_transpose_70341_70726 = 0x0;
const uint32_t sub_0012_address_model_135_tf_compat_v1_transpose_287_transpose_70343_70720 = 0x16a80;
const uint32_t sub_0012_address_model_135_tf_compat_v1_transpose_291_transpose_70350 = 0x16a80;

