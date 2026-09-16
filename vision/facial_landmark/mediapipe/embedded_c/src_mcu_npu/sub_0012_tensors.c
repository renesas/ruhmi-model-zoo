#include "sub_0012_tensors.h"

const TensorInfo sub_0012_tensors[] = {
  { "_split_1_command_stream", 0, 960, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 22768, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 129024, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 129024, "FAST_SCRATCH", 0x0 },
  { "p_re_lu_5_70139", 5, 73728, "INPUT_TENSOR", 0x9000 },
  { "add_5_70144", 4, 36864, "OUTPUT_TENSOR", 0x12000 },
};

const size_t sub_0012_tensors_count = sizeof(sub_0012_tensors) / sizeof(sub_0012_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0012_address_p_re_lu_5_70139 = 0x9000;
const uint32_t sub_0012_address_add_5_70144 = 0x12000;

