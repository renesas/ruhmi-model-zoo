#include "sub_0032_tensors.h"

const TensorInfo sub_0032_tensors[] = {
  { "_split_1_command_stream", 0, 748, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 26560, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 3744, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 3744, "FAST_SCRATCH", 0x0 },
  { "p_re_lu_25_70197", 7, 1152, "INPUT_TENSOR", 0x5a0 },
  { "p_re_lu_15_70185", 6, 1152, "INPUT_TENSOR", 0x0 },
  { "conv2d_28_70199", 5, 288, "OUTPUT_TENSOR", 0x480 },
  { "add_15_70188", 4, 1152, "OUTPUT_TENSOR", 0x5a0 },
};

const size_t sub_0032_tensors_count = sizeof(sub_0032_tensors) / sizeof(sub_0032_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0032_address_p_re_lu_25_70197 = 0x5a0;
const uint32_t sub_0032_address_p_re_lu_15_70185 = 0x0;
const uint32_t sub_0032_address_conv2d_28_70199 = 0x480;
const uint32_t sub_0032_address_add_15_70188 = 0x5a0;

