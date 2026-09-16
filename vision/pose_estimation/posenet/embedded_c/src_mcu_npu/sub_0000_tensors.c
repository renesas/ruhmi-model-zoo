#include "sub_0000_tensors.h"

const TensorInfo sub_0000_tensors[] = {
  { "_split_1_command_stream", 4, 3168, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 5, 618128, "MODEL", 0xffffffff },
  { "_split_1_scratch", 6, 798768, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 7, 798768, "FAST_SCRATCH", 0x0 },
  { "serving_default_input_0", 8, 198147, "INPUT_TENSOR", 0x0 },
  { "PartitionedCall_3_70093", 3, 9248, "OUTPUT_TENSOR", 0x12100 },
  { "PartitionedCall_2_70092", 2, 9248, "OUTPUT_TENSOR", 0x14520 },
  { "PartitionedCall_1_70091", 1, 9826, "OUTPUT_TENSOR", 0x16940 },
  { "PartitionedCall_0_70090", 0, 4913, "OUTPUT_TENSOR", 0x18fb0 },
};

const size_t sub_0000_tensors_count = sizeof(sub_0000_tensors) / sizeof(sub_0000_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0000_address_serving_default_input_0 = 0x0;
const uint32_t sub_0000_address_PartitionedCall_3_70093 = 0x12100;
const uint32_t sub_0000_address_PartitionedCall_2_70092 = 0x14520;
const uint32_t sub_0000_address_PartitionedCall_1_70091 = 0x16940;
const uint32_t sub_0000_address_PartitionedCall_0_70090 = 0x18fb0;

