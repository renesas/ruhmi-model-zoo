#include "sub_0024_tensors.h"

const TensorInfo sub_0024_tensors[] = {
  { "_split_1_command_stream", 0, 776, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 22160, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 27648, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 27648, "FAST_SCRATCH", 0x0 },
  { "p_re_lu_11_70167", 5, 18432, "INPUT_TENSOR", 0x0 },
  { "add_11_70171", 4, 4608, "OUTPUT_TENSOR", 0x0 },
};

const size_t sub_0024_tensors_count = sizeof(sub_0024_tensors) / sizeof(sub_0024_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0024_address_p_re_lu_11_70167 = 0x0;
const uint32_t sub_0024_address_add_11_70171 = 0x0;

