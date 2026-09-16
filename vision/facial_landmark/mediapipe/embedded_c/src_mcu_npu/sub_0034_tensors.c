#include "sub_0034_tensors.h"

const TensorInfo sub_0034_tensors[] = {
  { "_split_1_command_stream", 0, 968, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 24768, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 3744, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 3744, "FAST_SCRATCH", 0x0 },
  { "p_re_lu_26_70201", 7, 288, "INPUT_TENSOR", 0x5a0 },
  { "p_re_lu_16_70189", 6, 1152, "INPUT_TENSOR", 0x0 },
  { "add_23_70207", 5, 288, "OUTPUT_TENSOR", 0x480 },
  { "add_16_70196", 4, 1152, "OUTPUT_TENSOR", 0x5a0 },
};

const size_t sub_0034_tensors_count = sizeof(sub_0034_tensors) / sizeof(sub_0034_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0034_address_p_re_lu_26_70201 = 0x5a0;
const uint32_t sub_0034_address_p_re_lu_16_70189 = 0x0;
const uint32_t sub_0034_address_add_23_70207 = 0x480;
const uint32_t sub_0034_address_add_16_70196 = 0x5a0;

