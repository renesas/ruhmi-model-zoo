#include "sub_0038_tensors.h"

const TensorInfo sub_0038_tensors[] = {
  { "_split_1_command_stream", 0, 608, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 2672, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 864, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 864, "FAST_SCRATCH", 0x0 },
  { "p_re_lu_18_70202", 5, 288, "INPUT_TENSOR", 0x0 },
  { "add_17_70208", 4, 288, "OUTPUT_TENSOR", 0x120 },
};

const size_t sub_0038_tensors_count = sizeof(sub_0038_tensors) / sizeof(sub_0038_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0038_address_p_re_lu_18_70202 = 0x0;
const uint32_t sub_0038_address_add_17_70208 = 0x120;

