#include "sub_0016_tensors.h"

const TensorInfo sub_0016_tensors[] = {
  { "_split_1_command_stream", 0, 636, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 7296, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 110592, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 110592, "FAST_SCRATCH", 0x0 },
  { "p_re_lu_7_70149", 5, 36864, "INPUT_TENSOR", 0x0 },
  { "add_7_70152", 4, 36864, "OUTPUT_TENSOR", 0x9000 },
};

const size_t sub_0016_tensors_count = sizeof(sub_0016_tensors) / sizeof(sub_0016_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0016_address_p_re_lu_7_70149 = 0x0;
const uint32_t sub_0016_address_add_7_70152 = 0x9000;

