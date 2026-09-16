#include "sub_0010_tensors.h"

const TensorInfo sub_0010_tensors[] = {
  { "_split_1_command_stream", 0, 632, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 2656, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 221184, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 221184, "FAST_SCRATCH", 0x0 },
  { "p_re_lu_4_70135", 5, 73728, "INPUT_TENSOR", 0x0 },
  { "add_4_70138", 4, 73728, "OUTPUT_TENSOR", 0x12000 },
};

const size_t sub_0010_tensors_count = sizeof(sub_0010_tensors) / sizeof(sub_0010_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0010_address_p_re_lu_4_70135 = 0x0;
const uint32_t sub_0010_address_add_4_70138 = 0x12000;

