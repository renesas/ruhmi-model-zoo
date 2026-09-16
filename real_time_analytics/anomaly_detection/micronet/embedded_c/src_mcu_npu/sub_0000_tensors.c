#include "sub_0000_tensors.h"

const TensorInfo sub_0000_tensors[] = {
  { "_split_1_command_stream", 1, 1632, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 2, 473296, "MODEL", 0xffffffff },
  { "_split_1_scratch", 3, 245760, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 4, 245760, "FAST_SCRATCH", 0x0 },
  { "input", 5, 1024, "INPUT_TENSOR", 0x30000 },
  { "Identity_70038", 0, 8, "OUTPUT_TENSOR", 0x0 },
};

const size_t sub_0000_tensors_count = sizeof(sub_0000_tensors) / sizeof(sub_0000_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0000_address_input = 0x30000;
const uint32_t sub_0000_address_Identity_70038 = 0x0;

