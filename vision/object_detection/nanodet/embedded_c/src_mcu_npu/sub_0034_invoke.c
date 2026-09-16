#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "common_data.h"

#include "sub_0034_tensors.h"
#include "sub_0034_command_stream.h"
#include "sub_0034_model_data.h"

#include "sub_0034_invoke.h"

// Include Ethos-U driver headers (Assumed to be available)
#include "ethosu_driver.h"

// Define arenas with allocation and 16-byte alignment
__attribute__((aligned(16), section(".sdram"))) uint8_t sub_0034_arena[476000];
// Fast scratch arena not used for Ethos-U55
//  We will not create it for now and reuse the address of the other arena
// __attribute__((aligned(16))) static uint8_t sub_0034_fast_scratch[476000];
uint8_t* sub_0034_fast_scratch = sub_0034_arena;

int sub_0034_invoke(bool clean_outputs) {
  // Initialize base addresses and sizes
  uint64_t base_addrs[12] = {0};
  size_t base_addrs_size[12] = {0};
  int num_base_addrs = 12;

  // Variables for command stream
  uint8_t* cms_data = NULL;
  int cms_size = 0;

  // Prepare base_addrs and base_addrs_size arrays
  // Buffer sub_0034_model with size 400 and address: 4294967295
  base_addrs[0] = (uint64_t)(uintptr_t)sub_0034_model_data;
  base_addrs_size[0] = sub_0034_model_data_size;
  // Buffer sub_0034_arena with size 476000 and address: 0
  base_addrs[1] = (uint64_t)(uintptr_t) (sub_0034_arena+0);
  base_addrs_size[1] = 476000;

  // Buffer sub_0034_fast_scratch with size 476000 and address: 0
  base_addrs[2] = (uint64_t)(uintptr_t) (sub_0034_arena+0);
  base_addrs_size[2] = 476000;

  // Buffer input_tensor_0 with size 128000 and address: 296000
  base_addrs[3] = (uint64_t)(uintptr_t) (sub_0034_arena+296000);
  base_addrs_size[3] = 128000;

  // Buffer input_tensor_1 with size 51200 and address: 179200
  base_addrs[4] = (uint64_t)(uintptr_t) (sub_0034_arena+179200);
  base_addrs_size[4] = 51200;

  // Buffer input_tensor_2 with size 32000 and address: 264000
  base_addrs[5] = (uint64_t)(uintptr_t) (sub_0034_arena+264000);
  base_addrs_size[5] = 32000;

  // Buffer input_tensor_3 with size 12800 and address: 251200
  base_addrs[6] = (uint64_t)(uintptr_t) (sub_0034_arena+251200);
  base_addrs_size[6] = 12800;

  // Buffer input_tensor_4 with size 8000 and address: 238000
  base_addrs[7] = (uint64_t)(uintptr_t) (sub_0034_arena+238000);
  base_addrs_size[7] = 8000;

  // Buffer input_tensor_5 with size 3200 and address: 246000
  base_addrs[8] = (uint64_t)(uintptr_t) (sub_0034_arena+246000);
  base_addrs_size[8] = 3200;

  // Buffer input_tensor_6 with size 2000 and address: 249200
  base_addrs[9] = (uint64_t)(uintptr_t) (sub_0034_arena+249200);
  base_addrs_size[9] = 2000;

  // Buffer input_tensor_7 with size 800 and address: 475200
  base_addrs[10] = (uint64_t)(uintptr_t) (sub_0034_arena+475200);
  base_addrs_size[10] = 800;

  // Buffer output_tensor_0 with size 238000 and address: 238000
  if (clean_outputs) {
    memset(sub_0034_arena + 238000, 0, 238000);
  }
  base_addrs[11] = (uint64_t)(uintptr_t) (sub_0034_arena+238000);
  base_addrs_size[11] = 238000;

  // Command stream data
  cms_data = (uint8_t*)sub_0034_command_stream;
  cms_size = (int) sub_0034_command_stream_size;

  // Invoke the Ethos-U driver
  if (num_base_addrs > 8) {
    num_base_addrs = 8;
  }
  int result = ethosu_invoke_v3(&g_ethosu0, cms_data, cms_size, base_addrs, base_addrs_size, num_base_addrs, NULL);

  if (result == -1) {
    // Ethos-U invocation failed
    return -1;
  }

  return 0;
}
