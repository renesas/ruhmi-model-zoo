#include "sub_0000_tensors.h"

const TensorInfo sub_0000_tensors[] = {
  { "_split_1_command_stream", 0, 328, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 880, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 258048, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 258048, "FAST_SCRATCH", 0x0 },
  { "input_1", 5, 110592, "INPUT_TENSOR", 0x0 },
  { "conv2d_70116", 4, 147456, "OUTPUT_TENSOR", 0x1b000 },
};

const size_t sub_0000_tensors_count = sizeof(sub_0000_tensors) / sizeof(sub_0000_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0000_address_input_1 = 0x0;
const uint32_t sub_0000_address_conv2d_70116 = 0x1b000;

