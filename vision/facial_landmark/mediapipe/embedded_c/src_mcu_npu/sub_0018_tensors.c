#include "sub_0018_tensors.h"

const TensorInfo sub_0018_tensors[] = {
  { "_split_1_command_stream", 0, 936, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 21824, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 64512, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 64512, "FAST_SCRATCH", 0x0 },
  { "p_re_lu_8_70153", 5, 36864, "INPUT_TENSOR", 0x4800 },
  { "add_8_70158", 4, 18432, "OUTPUT_TENSOR", 0x9000 },
};

const size_t sub_0018_tensors_count = sizeof(sub_0018_tensors) / sizeof(sub_0018_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0018_address_p_re_lu_8_70153 = 0x4800;
const uint32_t sub_0018_address_add_8_70158 = 0x9000;

