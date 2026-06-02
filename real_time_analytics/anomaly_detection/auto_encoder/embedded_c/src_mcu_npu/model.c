/*
 * This file is developed by EdgeCortix Inc. to be used with certain Renesas Electronics Hardware only.
 *
 * Copyright © 2025 EdgeCortix Inc. Licensed to Renesas Electronics Corporation with the
 * right to sublicense under the Apache License, Version 2.0.
 *
 * This file also includes source code originally developed by the Renesas Electronics Corporation.
 * The Renesas disclaimer below applies to any Renesas-originated portions for usage of the code.
 *
 * The Renesas Electronics Corporation
 * DISCLAIMER
 * This software is supplied by Renesas Electronics Corporation and is only intended for use with Renesas products. No
 * other uses are authorized. This software is owned by Renesas Electronics Corporation and is protected under all
 * applicable laws, including copyright laws.
 * THIS SOFTWARE IS PROVIDED 'AS IS' AND RENESAS MAKES NO WARRANTIES REGARDING
 * THIS SOFTWARE, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. ALL SUCH WARRANTIES ARE EXPRESSLY DISCLAIMED. TO THE MAXIMUM
 * EXTENT PERMITTED NOT PROHIBITED BY LAW, NEITHER RENESAS ELECTRONICS CORPORATION NOR ANY OF ITS AFFILIATED COMPANIES
 * SHALL BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES FOR ANY REASON RELATED TO THIS
 * SOFTWARE, EVEN IF RENESAS OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
 * Renesas reserves the right, without notice, to make changes to this software and to discontinue the availability of
 * this software. By using this software, you agree to the additional terms and conditions found by accessing the
 * following link:
 * http://www.renesas.com/disclaimer
 *
 * Changed from original python code to C source code.
 * Copyright (C) 2017 Renesas Electronics Corporation. All rights reserved.
 *
 * This file also includes source codes originally developed by the TensorFlow Authors which were distributed under the following conditions.
 *
 * The TensorFlow Authors
 * Copyright 2023 The Apache Software Foundation
 *
 * This product includes software developed at
 * The Apache Software Foundation (http://www.apache.org/).
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "hal_data.h"

#include "model.h"

// CPU compute declarations
#include "compute_sub_0000.h"
#include "sub_0001_invoke.h"
#include "compute_sub_0002.h"

// Buffers for CPU units
float buf_input_1[640];
int8_t buf_input_1_10113[640];
int8_t buf_Identity_70029_10071[640];
float buf_Identity_70029[640];

// Arenas for CPU units
uint8_t compute_arena_sub_0000[kBufferSize_sub_0000];
uint8_t compute_arena_sub_0002[kBufferSize_sub_0002];

  // Model input pointers
float* GetModelInputPtr_input_1() {
  return buf_input_1;
}


  // Model output pointers
float* GetModelOutputPtr_Identity_70029() {
  return buf_Identity_70029;
}


void RunModel(bool clean_outputs) {
// CPU Unit
  compute_sub_0000(compute_arena_sub_0000, buf_input_1, buf_input_1_10113  );

  // Copy quantized input to NPU input tensor in arena
  memcpy(sub_0001_arena + sub_0001_address_input_1_10113,
         buf_input_1_10113,
         sizeof(buf_input_1_10113));

#if defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U)
  SCB_CleanDCache_by_Addr((uint32_t *)(void *)sub_0001_arena, (int32_t)kArenaSize_sub_0001);
#endif

// NPU Unit
  sub_0001_invoke(clean_outputs);

#if defined(__DCACHE_PRESENT) && (__DCACHE_PRESENT == 1U)
  SCB_InvalidateDCache_by_Addr((uint32_t *)(void *)sub_0001_arena, (int32_t)kArenaSize_sub_0001);
#endif

  // Copy NPU int8 output tensor from arena for CPU post stage
  memcpy(buf_Identity_70029_10071,
         sub_0001_arena + sub_0001_address_Identity_70029_10071,
         sizeof(buf_Identity_70029_10071));

// CPU Unit
  compute_sub_0002(compute_arena_sub_0002, buf_Identity_70029_10071, buf_Identity_70029  );
}
