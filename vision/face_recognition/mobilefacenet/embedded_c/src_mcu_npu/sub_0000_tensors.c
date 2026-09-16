#include "sub_0000_tensors.h"

const TensorInfo sub_0000_tensors[] = {
  { "_split_1_command_stream", 1, 6580, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 2, 1150416, "MODEL", 0xffffffff },
  { "_split_1_scratch", 3, 602112, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 4, 602112, "FAST_SCRATCH", 0x0 },
  { "serving_default_input0_0", 5, 37632, "INPUT_TENSOR", 0x31000 },
  { "PartitionedCall_0_70171", 0, 128, "OUTPUT_TENSOR", 0x200 },
};

const size_t sub_0000_tensors_count = sizeof(sub_0000_tensors) / sizeof(sub_0000_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0000_address_serving_default_input0_0 = 0x31000;
const uint32_t sub_0000_address_PartitionedCall_0_70171 = 0x200;

