#include "sub_0001_tensors.h"

const TensorInfo sub_0001_tensors[] = {
  { "_split_1_command_stream", 1, 3368, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 2, 229184, "MODEL", 0xffffffff },
  { "_split_1_scratch", 3, 98304, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 4, 98304, "FAST_SCRATCH", 0x0 },
  { "serving_default_input_1_0_10052", 5, 3072, "INPUT_TENSOR", 0x0 },
  { "StatefulPartitionedCall_1_0_70041_10128", 0, 10, "OUTPUT_TENSOR", 0x0 },
};

const size_t sub_0001_tensors_count = sizeof(sub_0001_tensors) / sizeof(sub_0001_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0001_address_serving_default_input_1_0_10052 = 0x0;
const uint32_t sub_0001_address_StatefulPartitionedCall_1_0_70041_10128 = 0x0;

