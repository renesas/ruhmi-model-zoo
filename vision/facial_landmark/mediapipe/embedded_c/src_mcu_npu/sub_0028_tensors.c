#include "sub_0028_tensors.h"

const TensorInfo sub_0028_tensors[] = {
  { "_split_1_command_stream", 0, 632, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 21936, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 13824, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 13824, "FAST_SCRATCH", 0x0 },
  { "p_re_lu_13_70176", 5, 4608, "INPUT_TENSOR", 0x0 },
  { "add_13_70179", 4, 4608, "OUTPUT_TENSOR", 0x1200 },
};

const size_t sub_0028_tensors_count = sizeof(sub_0028_tensors) / sizeof(sub_0028_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0028_address_p_re_lu_13_70176 = 0x0;
const uint32_t sub_0028_address_add_13_70179 = 0x1200;

