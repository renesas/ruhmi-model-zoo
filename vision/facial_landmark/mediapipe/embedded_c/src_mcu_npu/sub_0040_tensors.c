#include "sub_0040_tensors.h"

const TensorInfo sub_0040_tensors[] = {
  { "_split_1_command_stream", 0, 328, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 414416, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 1696, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 1696, "FAST_SCRATCH", 0x0 },
  { "p_re_lu_19_70210", 5, 288, "INPUT_TENSOR", 0x0 },
  { "conv2d_20_70212", 4, 1404, "OUTPUT_TENSOR", 0x120 },
};

const size_t sub_0040_tensors_count = sizeof(sub_0040_tensors) / sizeof(sub_0040_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0040_address_p_re_lu_19_70210 = 0x0;
const uint32_t sub_0040_address_conv2d_20_70212 = 0x120;

