#include "sub_0000_tensors.h"

const TensorInfo sub_0000_tensors[] = {
  { "_split_1_command_stream", 1, 6004, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 2, 867072, "MODEL", 0xffffffff },
  { "_split_1_scratch", 3, 46080, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 4, 46080, "FAST_SCRATCH", 0x0 },
  { "serving_default_input_0", 5, 6912, "INPUT_TENSOR", 0x4800 },
  { "PartitionedCall_0_70154", 0, 1000, "OUTPUT_TENSOR", 0xc0 },
};

const size_t sub_0000_tensors_count = sizeof(sub_0000_tensors) / sizeof(sub_0000_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0000_address_serving_default_input_0 = 0x4800;
const uint32_t sub_0000_address_PartitionedCall_0_70154 = 0xc0;

