#include "sub_0000_tensors.h"

const TensorInfo sub_0000_tensors[] = {
  { "_split_1_command_stream", 1, 2916, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 2, 36800, "MODEL", 0xffffffff },
  { "_split_1_scratch", 3, 16000, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 4, 16000, "FAST_SCRATCH", 0x0 },
  { "serving_default_input_1_0", 5, 490, "INPUT_TENSOR", 0x1f40 },
  { "StatefulPartitionedCall_0_70033", 0, 12, "OUTPUT_TENSOR", 0x0 },
};

const size_t sub_0000_tensors_count = sizeof(sub_0000_tensors) / sizeof(sub_0000_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0000_address_serving_default_input_1_0 = 0x1f40;
const uint32_t sub_0000_address_StatefulPartitionedCall_0_70033 = 0x0;

