#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "common_data.h"

#include "sub_0032_tensors.h"
#include "sub_0032_command_stream.h"
#include "sub_0032_model_data.h"

#include "sub_0032_invoke.h"

// Include Ethos-U driver headers (Assumed to be available)
#include "ethosu_driver.h"

// Define arenas with allocation and 16-byte alignment
__attribute__((aligned(16), section(".sdram"))) uint8_t sub_0032_arena[662400];
// Fast scratch arena not used for Ethos-U55
//  We will not create it for now and reuse the address of the other arena
// __attribute__((aligned(16))) static uint8_t sub_0032_fast_scratch[662400];
uint8_t* sub_0032_fast_scratch = sub_0032_arena;

int sub_0032_invoke(bool clean_outputs) {
  // Initialize base addresses and sizes
  uint64_t base_addrs[10] = {0};
  size_t base_addrs_size[10] = {0};
  int num_base_addrs = 10;

  // Variables for command stream
  uint8_t* cms_data = NULL;
  int cms_size = 0;

  // Prepare base_addrs and base_addrs_size arrays
  // Buffer sub_0032_model with size 566144 and address: 4294967295
  base_addrs[0] = (uint64_t)(uintptr_t)sub_0032_model_data;
  base_addrs_size[0] = sub_0032_model_data_size;
  // Buffer sub_0032_arena with size 662400 and address: 0
  base_addrs[1] = (uint64_t)(uintptr_t) (sub_0032_arena+0);
  base_addrs_size[1] = 662400;

  // Buffer sub_0032_fast_scratch with size 662400 and address: 0
  base_addrs[2] = (uint64_t)(uintptr_t) (sub_0032_arena+0);
  base_addrs_size[2] = 662400;

  // Buffer input_tensor_0 with size 46400 and address: 201600
  base_addrs[3] = (uint64_t)(uintptr_t) (sub_0032_arena+201600);
  base_addrs_size[3] = 46400;

  // Buffer input_tensor_1 with size 38400 and address: 9600
  base_addrs[4] = (uint64_t)(uintptr_t) (sub_0032_arena+9600);
  base_addrs_size[4] = 38400;

  // Buffer input_tensor_2 with size 153600 and address: 48000
  base_addrs[5] = (uint64_t)(uintptr_t) (sub_0032_arena+48000);
  base_addrs_size[5] = 153600;

  // Buffer output_tensor_0 with size 2800 and address: 0
  if (clean_outputs) {
    memset(sub_0032_arena + 0, 0, 2800);
  }
  base_addrs[6] = (uint64_t)(uintptr_t) (sub_0032_arena+0);
  base_addrs_size[6] = 2800;

  // Buffer output_tensor_1 with size 11200 and address: 2800
  if (clean_outputs) {
    memset(sub_0032_arena + 2800, 0, 11200);
  }
  base_addrs[7] = (uint64_t)(uintptr_t) (sub_0032_arena+2800);
  base_addrs_size[7] = 11200;

  // Buffer output_tensor_2 with size 44800 and address: 14000
  if (clean_outputs) {
    memset(sub_0032_arena + 14000, 0, 44800);
  }
  base_addrs[8] = (uint64_t)(uintptr_t) (sub_0032_arena+14000);
  base_addrs_size[8] = 44800;

  // Buffer output_tensor_3 with size 179200 and address: 212400
  if (clean_outputs) {
    memset(sub_0032_arena + 212400, 0, 179200);
  }
  base_addrs[9] = (uint64_t)(uintptr_t) (sub_0032_arena+212400);
  base_addrs_size[9] = 179200;

  // Command stream data
  cms_data = (uint8_t*)sub_0032_command_stream;
  cms_size = (int) sub_0032_command_stream_size;

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
