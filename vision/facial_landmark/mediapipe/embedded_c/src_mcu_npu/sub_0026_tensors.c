#include "sub_0026_tensors.h"

const TensorInfo sub_0026_tensors[] = {
  { "_split_1_command_stream", 0, 632, "COMMAND_STREAM", 0xffffffff },
  { "_split_1_flash", 1, 21936, "MODEL", 0xffffffff },
  { "_split_1_scratch", 2, 13824, "ARENA", 0x0 },
  { "_split_1_scratch_fast", 3, 13824, "FAST_SCRATCH", 0x0 },
  { "p_re_lu_12_70172", 5, 4608, "INPUT_TENSOR", 0x0 },
  { "add_12_70175", 4, 4608, "OUTPUT_TENSOR", 0x1200 },
};

const size_t sub_0026_tensors_count = sizeof(sub_0026_tensors) / sizeof(sub_0026_tensors[0]);

// Addresses for each input and output buffer inside of the arena
const uint32_t sub_0026_address_p_re_lu_12_70172 = 0x0;
const uint32_t sub_0026_address_add_12_70175 = 0x1200;

