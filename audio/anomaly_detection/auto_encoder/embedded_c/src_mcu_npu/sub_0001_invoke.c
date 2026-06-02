#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "common_data.h"

#include "sub_0001_tensors.h"
#include "sub_0001_command_stream.h"
#include "sub_0001_model_data.h"

#include "sub_0001_invoke.h"

// Include Ethos-U driver headers (Assumed to be available)
#include "ethosu_driver.h"

// Define arenas with allocation and 16-byte alignment
__attribute__((aligned(16))) uint8_t sub_0001_arena[768];
// Fast scratch arena not used for Ethos-U55
//  We will not create it for now and reuse the address of the other arena
// __attribute__((aligned(16))) static uint8_t sub_0001_fast_scratch[768];
uint8_t* sub_0001_fast_scratch = sub_0001_arena;

int sub_0001_invoke(bool clean_outputs) {
  // Initialize base addresses and sizes
  uint64_t base_addrs[5] = {0};
  size_t base_addrs_size[5] = {0};
  int num_base_addrs = 5;

  // Variables for command stream
  uint8_t* cms_data = NULL;
  int cms_size = 0;

  // Prepare base_addrs and base_addrs_size arrays
  // Buffer sub_0001_model with size 217968 and address: 4294967295
  base_addrs[0] = (uint64_t)(uintptr_t)sub_0001_model_data;
  base_addrs_size[0] = sub_0001_model_data_size;
  // Buffer sub_0001_arena with size 768 and address: 0
  base_addrs[1] = (uint64_t)(uintptr_t) (sub_0001_arena+0);
  base_addrs_size[1] = 768;

  // Buffer sub_0001_fast_scratch with size 768 and address: 0
  base_addrs[2] = (uint64_t)(uintptr_t) (sub_0001_arena+0);
  base_addrs_size[2] = 768;

  // Buffer input_tensor_0 with size 640 and address: 0
  base_addrs[3] = (uint64_t)(uintptr_t) (sub_0001_arena+0);
  base_addrs_size[3] = 640;

  // Buffer output_tensor_0 with size 640 and address: 128
  if (clean_outputs) {
    const size_t input_start = 0U;
    const size_t input_size = 640U;
    const size_t output_start = 128U;
    const size_t output_size = 640U;
    const size_t input_end = input_start + input_size;
    const size_t output_end = output_start + output_size;
    const bool input_output_overlap = !((output_end <= input_start) || (input_end <= output_start));

    // Do not clear output region when it overlaps the input region.
    // For this graph, output starts at 0x80 and overlaps input at 0x00.
    if (!input_output_overlap) {
      memset(sub_0001_arena + output_start, 0, output_size);
    }
  }
  base_addrs[4] = (uint64_t)(uintptr_t) (sub_0001_arena+128);
  base_addrs_size[4] = 640;

  // Command stream data
  cms_data = (uint8_t*)sub_0001_command_stream;
  cms_size = (int) sub_0001_command_stream_size;

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
