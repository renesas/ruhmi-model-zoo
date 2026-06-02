#include "sub_0000_tensors.h"

const TensorInfo sub_0000_tensors[] = {
  { "_split_1_command_stream", 4, 8096, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 5, 175200, "MODEL", 0xffffffff },
  { "_split_1_scratch", 6, 393216, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 7, 393216, "FAST_SCRATCH", 0x0 },
  { "serving_default_input_0", 8, 49152, "INPUT_TENSOR", 0x0 },
  { "StatefulPartitionedCall_3_70151", 3, 6144, "OUTPUT_TENSOR", 0x6000 },
  { "StatefulPartitionedCall_1_70149", 1, 384, "OUTPUT_TENSOR", 0x9000 },
  { "StatefulPartitionedCall_2_70130", 2, 8192, "OUTPUT_TENSOR", 0x9180 },
  { "StatefulPartitionedCall_0_70128", 0, 512, "OUTPUT_TENSOR", 0x7800 },
};

const size_t sub_0000_tensors_count = sizeof(sub_0000_tensors) / sizeof(sub_0000_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0000_address_serving_default_input_0 = 0x0;
const uint32_t sub_0000_address_StatefulPartitionedCall_3_70151 = 0x6000;
const uint32_t sub_0000_address_StatefulPartitionedCall_1_70149 = 0x9000;
const uint32_t sub_0000_address_StatefulPartitionedCall_2_70130 = 0x9180;
const uint32_t sub_0000_address_StatefulPartitionedCall_0_70128 = 0x7800;

