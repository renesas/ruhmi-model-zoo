#include "sub_0004_tensors.h"

const TensorInfo sub_0004_tensors[] = {
  { "_split_1_command_stream", 0, 604, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 1152, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 442368, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 442368, "FAST_SCRATCH", 0x0 },
  { "p_re_lu_1_70121", 5, 147456, "INPUT_TENSOR", 0x0 },
  { "add_1_70124", 4, 147456, "OUTPUT_TENSOR", 0x24000 },
};

const size_t sub_0004_tensors_count = sizeof(sub_0004_tensors) / sizeof(sub_0004_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0004_address_p_re_lu_1_70121 = 0x0;
const uint32_t sub_0004_address_add_1_70124 = 0x24000;

