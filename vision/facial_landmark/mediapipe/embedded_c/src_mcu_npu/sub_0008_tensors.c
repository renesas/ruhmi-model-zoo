#include "sub_0008_tensors.h"

const TensorInfo sub_0008_tensors[] = {
  { "_split_1_command_stream", 0, 636, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 2672, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 221184, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 221184, "FAST_SCRATCH", 0x0 },
  { "p_re_lu_3_70131", 5, 73728, "INPUT_TENSOR", 0x0 },
  { "add_3_70134", 4, 73728, "OUTPUT_TENSOR", 0x12000 },
};

const size_t sub_0008_tensors_count = sizeof(sub_0008_tensors) / sizeof(sub_0008_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0008_address_p_re_lu_3_70131 = 0x0;
const uint32_t sub_0008_address_add_3_70134 = 0x12000;

