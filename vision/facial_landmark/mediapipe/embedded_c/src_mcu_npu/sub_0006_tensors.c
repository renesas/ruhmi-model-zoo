#include "sub_0006_tensors.h"

const TensorInfo sub_0006_tensors[] = {
  { "_split_1_command_stream", 0, 912, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 38688, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 258048, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 258048, "FAST_SCRATCH", 0x0 },
  { "p_re_lu_2_70125", 5, 147456, "INPUT_TENSOR", 0x12000 },
  { "add_2_70130", 4, 73728, "OUTPUT_TENSOR", 0x24000 },
};

const size_t sub_0006_tensors_count = sizeof(sub_0006_tensors) / sizeof(sub_0006_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0006_address_p_re_lu_2_70125 = 0x12000;
const uint32_t sub_0006_address_add_2_70130 = 0x24000;

