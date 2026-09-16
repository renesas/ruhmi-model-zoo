#include "sub_0020_tensors.h"

const TensorInfo sub_0020_tensors[] = {
  { "_split_1_command_stream", 0, 1180, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 36016, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 195200, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 195200, "FAST_SCRATCH", 0x0 },
  { "model_135_tf_compat_v1_transpose_320_transpose_70397_70798", 5, 46400, "INPUT_TENSOR", 0x0 },
  { "model_135_tf_compat_v1_transpose_319_transpose_70399_70792", 4, 46400, "INPUT_TENSOR", 0x16a80 },
  { "model_135_tf_compat_v1_transpose_323_transpose_70406", 6, 92800, "OUTPUT_TENSOR", 0x16a80 },
};

const size_t sub_0020_tensors_count = sizeof(sub_0020_tensors) / sizeof(sub_0020_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0020_address_model_135_tf_compat_v1_transpose_320_transpose_70397_70798 = 0x0;
const uint32_t sub_0020_address_model_135_tf_compat_v1_transpose_319_transpose_70399_70792 = 0x16a80;
const uint32_t sub_0020_address_model_135_tf_compat_v1_transpose_323_transpose_70406 = 0x16a80;

