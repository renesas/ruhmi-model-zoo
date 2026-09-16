#include "sub_0036_tensors.h"

const TensorInfo sub_0036_tensors[] = {
  { "_split_1_command_stream", 0, 476, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 5024, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 1456, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 1456, "FAST_SCRATCH", 0x0 },
  { "p_re_lu_27_70209", 7, 288, "INPUT_TENSOR", 0x490 },
  { "p_re_lu_17_70198", 6, 1152, "INPUT_TENSOR", 0x0 },
  { "conv2d_30_70211", 5, 1, "OUTPUT_TENSOR", 0x480 },
  { "conv2d_18_70200", 4, 288, "OUTPUT_TENSOR", 0x490 },
};

const size_t sub_0036_tensors_count = sizeof(sub_0036_tensors) / sizeof(sub_0036_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0036_address_p_re_lu_27_70209 = 0x490;
const uint32_t sub_0036_address_p_re_lu_17_70198 = 0x0;
const uint32_t sub_0036_address_conv2d_30_70211 = 0x480;
const uint32_t sub_0036_address_conv2d_18_70200 = 0x490;

