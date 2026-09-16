#include "sub_0001_tensors.h"

const TensorInfo sub_0001_tensors[] = {
  { "_split_1_command_stream", 1, 1220, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 2, 2395648, "MODEL", 0xffffffff },
  { "_split_1_scratch", 3, 222592, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 4, 222592, "FAST_SCRATCH", 0x0 },
  { "model_reshape_Reshape_70019", 5, 11544, "INPUT_TENSOR", 0x9400 },
  { "Identity_int8_70036", 0, 4292, "OUTPUT_TENSOR", 0x0 },
};

const size_t sub_0001_tensors_count = sizeof(sub_0001_tensors) / sizeof(sub_0001_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0001_address_model_reshape_Reshape_70019 = 0x9400;
const uint32_t sub_0001_address_Identity_int8_70036 = 0x0;

