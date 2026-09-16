#include "sub_0030_tensors.h"

const TensorInfo sub_0030_tensors[] = {
  { "_split_1_command_stream", 0, 1216, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 44304, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 8064, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 8064, "FAST_SCRATCH", 0x0 },
  { "p_re_lu_14_70180", 6, 4608, "INPUT_TENSOR", 0x0 },
  { "add_22_70195", 5, 1152, "OUTPUT_TENSOR", 0x1200 },
  { "add_14_70184", 4, 1152, "OUTPUT_TENSOR", 0x0 },
};

const size_t sub_0030_tensors_count = sizeof(sub_0030_tensors) / sizeof(sub_0030_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0030_address_p_re_lu_14_70180 = 0x0;
const uint32_t sub_0030_address_add_22_70195 = 0x1200;
const uint32_t sub_0030_address_add_14_70184 = 0x0;

