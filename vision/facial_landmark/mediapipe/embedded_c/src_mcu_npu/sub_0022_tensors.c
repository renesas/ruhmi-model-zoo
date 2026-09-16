#include "sub_0022_tensors.h"

const TensorInfo sub_0022_tensors[] = {
  { "_split_1_command_stream", 0, 636, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 22176, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 55296, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 55296, "FAST_SCRATCH", 0x0 },
  { "p_re_lu_10_70163", 5, 18432, "INPUT_TENSOR", 0x0 },
  { "add_10_70166", 4, 18432, "OUTPUT_TENSOR", 0x4800 },
};

const size_t sub_0022_tensors_count = sizeof(sub_0022_tensors) / sizeof(sub_0022_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0022_address_p_re_lu_10_70163 = 0x0;
const uint32_t sub_0022_address_add_10_70166 = 0x4800;

