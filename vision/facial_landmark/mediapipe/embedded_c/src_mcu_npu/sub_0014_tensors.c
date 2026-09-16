#include "sub_0014_tensors.h"

const TensorInfo sub_0014_tensors[] = {
  { "_split_1_command_stream", 0, 640, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 7296, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 110592, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 110592, "FAST_SCRATCH", 0x0 },
  { "p_re_lu_6_70145", 5, 36864, "INPUT_TENSOR", 0x0 },
  { "add_6_70148", 4, 36864, "OUTPUT_TENSOR", 0x9000 },
};

const size_t sub_0014_tensors_count = sizeof(sub_0014_tensors) / sizeof(sub_0014_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0014_address_p_re_lu_6_70145 = 0x0;
const uint32_t sub_0014_address_add_6_70148 = 0x9000;

