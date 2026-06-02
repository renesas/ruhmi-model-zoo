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
#include <string.h>
#include <stdbool.h>

/* Provides SCB_CleanDCache_by_Addr / SCB_InvalidateDCache_by_Addr (CMSIS-Core
 * via the Renesas FSP BSP) needed for CPU↔NPU D-cache coherency maintenance. */
#include "common_data.h"

#include "model.h"

// CPU compute declarations
#include "sub_0000_invoke.h"
#include "compute_sub_0001.h"
#include "sub_0002_invoke.h"
#include "compute_sub_0003.h"
#include "sub_0004_invoke.h"
#include "compute_sub_0005.h"
#include "sub_0006_invoke.h"
#include "compute_sub_0007.h"
#include "sub_0008_invoke.h"
#include "compute_sub_0009.h"
#include "sub_0010_invoke.h"
#include "compute_sub_0011.h"
#include "sub_0012_invoke.h"
#include "compute_sub_0013.h"
#include "sub_0014_invoke.h"
#include "compute_sub_0015.h"
#include "sub_0016_invoke.h"
#include "compute_sub_0017.h"
#include "sub_0018_invoke.h"
#include "compute_sub_0019.h"
#include "sub_0020_invoke.h"
#include "compute_sub_0021.h"
#include "sub_0022_invoke.h"
#include "compute_sub_0023.h"
#include "sub_0024_invoke.h"
#include "compute_sub_0025.h"
#include "sub_0026_invoke.h"
#include "compute_sub_0027.h"
#include "sub_0028_invoke.h"
#include "compute_sub_0029.h"
#include "sub_0030_invoke.h"
#include "compute_sub_0031.h"
#include "sub_0032_invoke.h"

// Buffers for CPU units
int8_t buf_serving_default_input_0[150528];
int8_t buf_model_125_tf_reshape_97_Reshape_70141[37632];
int8_t buf_model_125_tf_reshape_99_Reshape_70157[37632];
int8_t buf_model_125_tf_reshape_101_Reshape_70173[37632];
int8_t buf_model_125_tf_compat_v1_transpose_221_transpose_70190_70441[37632];
int8_t buf_model_125_tf_reshape_105_Reshape_70202[18816];
int8_t buf_model_125_tf_reshape_107_Reshape_70218[18816];
int8_t buf_model_125_tf_reshape_109_Reshape_70234[18816];
int8_t buf_model_125_tf_reshape_111_Reshape_70250[18816];
int8_t buf_model_125_tf_reshape_113_Reshape_70266[18816];
int8_t buf_model_125_tf_reshape_115_Reshape_70282[18816];
int8_t buf_model_125_tf_reshape_117_Reshape_70298[18816];
int8_t buf_model_125_tf_compat_v1_transpose_285_transpose_70315_70579[18816];
int8_t buf_model_125_tf_reshape_121_Reshape_70327[9408];
int8_t buf_model_125_tf_reshape_123_Reshape_70343[9408];
int8_t buf_model_125_tf_reshape_125_Reshape_70359[9408];
int8_t buf_model_125_tf_compat_v1_transpose_317_transpose_70376_70645[9408];

// Arenas for CPU units
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0001[kBufferSize_sub_0001];
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0003[kBufferSize_sub_0003];
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0005[kBufferSize_sub_0005];
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0007[kBufferSize_sub_0007];
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0009[kBufferSize_sub_0009];
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0011[kBufferSize_sub_0011];
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0013[kBufferSize_sub_0013];
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0015[kBufferSize_sub_0015];
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0017[kBufferSize_sub_0017];
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0019[kBufferSize_sub_0019];
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0021[kBufferSize_sub_0021];
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0023[kBufferSize_sub_0023];
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0025[kBufferSize_sub_0025];
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0027[kBufferSize_sub_0027];
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0029[kBufferSize_sub_0029];
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0031[kBufferSize_sub_0031];

  // Model input pointers
int8_t* GetModelInputPtr_serving_default_input_0() {
  return (int8_t*) (sub_0000_arena + sub_0000_address_serving_default_input_0);
}


  // Model output pointers
int8_t* GetModelOutputPtr_StatefulPartitionedCall_0_70379() {
  return (int8_t*) (sub_0032_arena + sub_0032_address_StatefulPartitionedCall_0_70379);
}


/* Returns 0 on success, or the (negative) index of the first failing NPU unit. */
int RunModel(bool clean_outputs) {
  int _npu_status = 0;
  // Buffers for NPU units
  int8_t* buf_model_125_tf_compat_v1_transpose_193_transpose_70138 = (int8_t*) (sub_0000_arena + sub_0000_address_model_125_tf_compat_v1_transpose_193_transpose_70138);
  int8_t* buf_model_125_tf_compat_v1_transpose_201_transpose_70154 = (int8_t*) (sub_0002_arena + sub_0002_address_model_125_tf_compat_v1_transpose_201_transpose_70154);
  int8_t* buf_model_125_tf_compat_v1_transpose_209_transpose_70170 = (int8_t*) (sub_0004_arena + sub_0004_address_model_125_tf_compat_v1_transpose_209_transpose_70170);
  int8_t* buf_model_125_tf_compat_v1_transpose_217_transpose_70186 = (int8_t*) (sub_0006_arena + sub_0006_address_model_125_tf_compat_v1_transpose_217_transpose_70186);
  int8_t* buf_model_125_tf_compat_v1_transpose_225_transpose_70199 = (int8_t*) (sub_0008_arena + sub_0008_address_model_125_tf_compat_v1_transpose_225_transpose_70199);
  int8_t* buf_model_125_tf_compat_v1_transpose_233_transpose_70215 = (int8_t*) (sub_0010_arena + sub_0010_address_model_125_tf_compat_v1_transpose_233_transpose_70215);
  int8_t* buf_model_125_tf_compat_v1_transpose_241_transpose_70231 = (int8_t*) (sub_0012_arena + sub_0012_address_model_125_tf_compat_v1_transpose_241_transpose_70231);
  int8_t* buf_model_125_tf_compat_v1_transpose_249_transpose_70247 = (int8_t*) (sub_0014_arena + sub_0014_address_model_125_tf_compat_v1_transpose_249_transpose_70247);
  int8_t* buf_model_125_tf_compat_v1_transpose_257_transpose_70263 = (int8_t*) (sub_0016_arena + sub_0016_address_model_125_tf_compat_v1_transpose_257_transpose_70263);
  int8_t* buf_model_125_tf_compat_v1_transpose_265_transpose_70279 = (int8_t*) (sub_0018_arena + sub_0018_address_model_125_tf_compat_v1_transpose_265_transpose_70279);
  int8_t* buf_model_125_tf_compat_v1_transpose_273_transpose_70295 = (int8_t*) (sub_0020_arena + sub_0020_address_model_125_tf_compat_v1_transpose_273_transpose_70295);
  int8_t* buf_model_125_tf_compat_v1_transpose_281_transpose_70311 = (int8_t*) (sub_0022_arena + sub_0022_address_model_125_tf_compat_v1_transpose_281_transpose_70311);
  int8_t* buf_model_125_tf_compat_v1_transpose_289_transpose_70324 = (int8_t*) (sub_0024_arena + sub_0024_address_model_125_tf_compat_v1_transpose_289_transpose_70324);
  int8_t* buf_model_125_tf_compat_v1_transpose_297_transpose_70340 = (int8_t*) (sub_0026_arena + sub_0026_address_model_125_tf_compat_v1_transpose_297_transpose_70340);
  int8_t* buf_model_125_tf_compat_v1_transpose_305_transpose_70356 = (int8_t*) (sub_0028_arena + sub_0028_address_model_125_tf_compat_v1_transpose_305_transpose_70356);
  int8_t* buf_model_125_tf_compat_v1_transpose_313_transpose_70372 = (int8_t*) (sub_0030_arena + sub_0030_address_model_125_tf_compat_v1_transpose_313_transpose_70372);
  int8_t* buf_StatefulPartitionedCall_0_70379 = (int8_t*) (sub_0032_arena + sub_0032_address_StatefulPartitionedCall_0_70379);
  // NPU Unit — sub_0000
  // Input already flushed by caller (ruhmi_perf_eval.c SCB_CleanDCache call).
  if (sub_0000_invoke(clean_outputs) != 0) { _npu_status = -1; }
  // NPU wrote output to sub_0000_arena[37632..75263]. Invalidate so CPU reads fresh SDRAM.
  SCB_InvalidateDCache_by_Addr((void *)(sub_0000_arena + 37632U), 37632);

  // CPU Unit
  compute_sub_0001(compute_arena_sub_0001, buf_model_125_tf_compat_v1_transpose_193_transpose_70138, buf_model_125_tf_reshape_97_Reshape_70141  );

  memcpy((sub_0002_arena + sub_0002_address_model_125_tf_reshape_97_Reshape_70141), buf_model_125_tf_reshape_97_Reshape_70141, 37632);
  // NPU Unit — sub_0002
  SCB_CleanDCache_by_Addr((void *)(sub_0002_arena + 56448U), 37632);
  if (sub_0002_invoke(clean_outputs) != 0) { if (_npu_status == 0) _npu_status = -2; }
  SCB_InvalidateDCache_by_Addr((void *)(sub_0002_arena + 37632U), 37632);

  // CPU Unit
  compute_sub_0003(compute_arena_sub_0003, buf_model_125_tf_compat_v1_transpose_201_transpose_70154, buf_model_125_tf_reshape_99_Reshape_70157  );

  memcpy((sub_0004_arena + sub_0004_address_model_125_tf_reshape_99_Reshape_70157), buf_model_125_tf_reshape_99_Reshape_70157, 37632);
  // NPU Unit — sub_0004
  SCB_CleanDCache_by_Addr((void *)(sub_0004_arena + 56448U), 37632);
  if (sub_0004_invoke(clean_outputs) != 0) { if (_npu_status == 0) _npu_status = -4; }
  SCB_InvalidateDCache_by_Addr((void *)(sub_0004_arena + 37632U), 37632);

  // CPU Unit
  compute_sub_0005(compute_arena_sub_0005, buf_model_125_tf_compat_v1_transpose_209_transpose_70170, buf_model_125_tf_reshape_101_Reshape_70173  );

  memcpy((sub_0006_arena + sub_0006_address_model_125_tf_reshape_101_Reshape_70173), buf_model_125_tf_reshape_101_Reshape_70173, 37632);
  // NPU Unit — sub_0006
  SCB_CleanDCache_by_Addr((void *)(sub_0006_arena + 56448U), 37632);
  if (sub_0006_invoke(clean_outputs) != 0) { if (_npu_status == 0) _npu_status = -6; }
  SCB_InvalidateDCache_by_Addr((void *)(sub_0006_arena + 37632U), 37632);

  // CPU Unit
  compute_sub_0007(compute_arena_sub_0007, buf_model_125_tf_compat_v1_transpose_217_transpose_70186, buf_model_125_tf_compat_v1_transpose_221_transpose_70190_70441  );

  memcpy((sub_0008_arena + sub_0008_address_model_125_tf_compat_v1_transpose_221_transpose_70190_70441), buf_model_125_tf_compat_v1_transpose_221_transpose_70190_70441, 37632);
  // NPU Unit — sub_0008  (input @37632, output @0)
  SCB_CleanDCache_by_Addr((void *)(sub_0008_arena + 37632U), 37632);
  if (sub_0008_invoke(clean_outputs) != 0) { if (_npu_status == 0) _npu_status = -8; }
  SCB_InvalidateDCache_by_Addr((void *)(sub_0008_arena + 0U), 18816);

  // CPU Unit
  compute_sub_0009(compute_arena_sub_0009, buf_model_125_tf_compat_v1_transpose_225_transpose_70199, buf_model_125_tf_reshape_105_Reshape_70202  );

  memcpy((sub_0010_arena + sub_0010_address_model_125_tf_reshape_105_Reshape_70202), buf_model_125_tf_reshape_105_Reshape_70202, 18816);
  // NPU Unit — sub_0010  (input @18816, output @18816)
  SCB_CleanDCache_by_Addr((void *)(sub_0010_arena + 18816U), 18816);
  if (sub_0010_invoke(clean_outputs) != 0) { if (_npu_status == 0) _npu_status = -10; }
  SCB_InvalidateDCache_by_Addr((void *)(sub_0010_arena + 18816U), 18816);

  // CPU Unit
  compute_sub_0011(compute_arena_sub_0011, buf_model_125_tf_compat_v1_transpose_233_transpose_70215, buf_model_125_tf_reshape_107_Reshape_70218  );

  memcpy((sub_0012_arena + sub_0012_address_model_125_tf_reshape_107_Reshape_70218), buf_model_125_tf_reshape_107_Reshape_70218, 18816);
  // NPU Unit — sub_0012  (input @18816, output @18816)
  SCB_CleanDCache_by_Addr((void *)(sub_0012_arena + 18816U), 18816);
  if (sub_0012_invoke(clean_outputs) != 0) { if (_npu_status == 0) _npu_status = -12; }
  SCB_InvalidateDCache_by_Addr((void *)(sub_0012_arena + 18816U), 18816);

  // CPU Unit
  compute_sub_0013(compute_arena_sub_0013, buf_model_125_tf_compat_v1_transpose_241_transpose_70231, buf_model_125_tf_reshape_109_Reshape_70234  );

  memcpy((sub_0014_arena + sub_0014_address_model_125_tf_reshape_109_Reshape_70234), buf_model_125_tf_reshape_109_Reshape_70234, 18816);
  // NPU Unit — sub_0014  (input @18816, output @18816)
  SCB_CleanDCache_by_Addr((void *)(sub_0014_arena + 18816U), 18816);
  if (sub_0014_invoke(clean_outputs) != 0) { if (_npu_status == 0) _npu_status = -14; }
  SCB_InvalidateDCache_by_Addr((void *)(sub_0014_arena + 18816U), 18816);

  // CPU Unit
  compute_sub_0015(compute_arena_sub_0015, buf_model_125_tf_compat_v1_transpose_249_transpose_70247, buf_model_125_tf_reshape_111_Reshape_70250  );

  memcpy((sub_0016_arena + sub_0016_address_model_125_tf_reshape_111_Reshape_70250), buf_model_125_tf_reshape_111_Reshape_70250, 18816);
  // NPU Unit — sub_0016  (input @18816, output @18816)
  SCB_CleanDCache_by_Addr((void *)(sub_0016_arena + 18816U), 18816);
  if (sub_0016_invoke(clean_outputs) != 0) { if (_npu_status == 0) _npu_status = -16; }
  SCB_InvalidateDCache_by_Addr((void *)(sub_0016_arena + 18816U), 18816);

  // CPU Unit
  compute_sub_0017(compute_arena_sub_0017, buf_model_125_tf_compat_v1_transpose_257_transpose_70263, buf_model_125_tf_reshape_113_Reshape_70266  );

  memcpy((sub_0018_arena + sub_0018_address_model_125_tf_reshape_113_Reshape_70266), buf_model_125_tf_reshape_113_Reshape_70266, 18816);
  // NPU Unit — sub_0018  (input @18816, output @18816)
  SCB_CleanDCache_by_Addr((void *)(sub_0018_arena + 18816U), 18816);
  if (sub_0018_invoke(clean_outputs) != 0) { if (_npu_status == 0) _npu_status = -18; }
  SCB_InvalidateDCache_by_Addr((void *)(sub_0018_arena + 18816U), 18816);

  // CPU Unit
  compute_sub_0019(compute_arena_sub_0019, buf_model_125_tf_compat_v1_transpose_265_transpose_70279, buf_model_125_tf_reshape_115_Reshape_70282  );

  memcpy((sub_0020_arena + sub_0020_address_model_125_tf_reshape_115_Reshape_70282), buf_model_125_tf_reshape_115_Reshape_70282, 18816);
  // NPU Unit — sub_0020  (input @18816, output @18816)
  SCB_CleanDCache_by_Addr((void *)(sub_0020_arena + 18816U), 18816);
  if (sub_0020_invoke(clean_outputs) != 0) { if (_npu_status == 0) _npu_status = -20; }
  SCB_InvalidateDCache_by_Addr((void *)(sub_0020_arena + 18816U), 18816);

  // CPU Unit
  compute_sub_0021(compute_arena_sub_0021, buf_model_125_tf_compat_v1_transpose_273_transpose_70295, buf_model_125_tf_reshape_117_Reshape_70298  );

  memcpy((sub_0022_arena + sub_0022_address_model_125_tf_reshape_117_Reshape_70298), buf_model_125_tf_reshape_117_Reshape_70298, 18816);
  // NPU Unit — sub_0022  (input @18816, output @18816)
  SCB_CleanDCache_by_Addr((void *)(sub_0022_arena + 18816U), 18816);
  if (sub_0022_invoke(clean_outputs) != 0) { if (_npu_status == 0) _npu_status = -22; }
  SCB_InvalidateDCache_by_Addr((void *)(sub_0022_arena + 18816U), 18816);

  // CPU Unit
  compute_sub_0023(compute_arena_sub_0023, buf_model_125_tf_compat_v1_transpose_281_transpose_70311, buf_model_125_tf_compat_v1_transpose_285_transpose_70315_70579  );

  memcpy((sub_0024_arena + sub_0024_address_model_125_tf_compat_v1_transpose_285_transpose_70315_70579), buf_model_125_tf_compat_v1_transpose_285_transpose_70315_70579, 18816);
  // NPU Unit — sub_0024  (input @18816, output @0)
  SCB_CleanDCache_by_Addr((void *)(sub_0024_arena + 18816U), 18816);
  if (sub_0024_invoke(clean_outputs) != 0) { if (_npu_status == 0) _npu_status = -24; }
  SCB_InvalidateDCache_by_Addr((void *)(sub_0024_arena + 0U), 9408);

  // CPU Unit
  compute_sub_0025(compute_arena_sub_0025, buf_model_125_tf_compat_v1_transpose_289_transpose_70324, buf_model_125_tf_reshape_121_Reshape_70327  );

  memcpy((sub_0026_arena + sub_0026_address_model_125_tf_reshape_121_Reshape_70327), buf_model_125_tf_reshape_121_Reshape_70327, 9408);
  // NPU Unit — sub_0026  (input @9408, output @9408)
  SCB_CleanDCache_by_Addr((void *)(sub_0026_arena + 9408U), 9408);
  if (sub_0026_invoke(clean_outputs) != 0) { if (_npu_status == 0) _npu_status = -26; }
  SCB_InvalidateDCache_by_Addr((void *)(sub_0026_arena + 9408U), 9408);

  // CPU Unit
  compute_sub_0027(compute_arena_sub_0027, buf_model_125_tf_compat_v1_transpose_297_transpose_70340, buf_model_125_tf_reshape_123_Reshape_70343  );

  memcpy((sub_0028_arena + sub_0028_address_model_125_tf_reshape_123_Reshape_70343), buf_model_125_tf_reshape_123_Reshape_70343, 9408);
  // NPU Unit — sub_0028  (input @9408, output @9408)
  SCB_CleanDCache_by_Addr((void *)(sub_0028_arena + 9408U), 9408);
  if (sub_0028_invoke(clean_outputs) != 0) { if (_npu_status == 0) _npu_status = -28; }
  SCB_InvalidateDCache_by_Addr((void *)(sub_0028_arena + 9408U), 9408);

  // CPU Unit
  compute_sub_0029(compute_arena_sub_0029, buf_model_125_tf_compat_v1_transpose_305_transpose_70356, buf_model_125_tf_reshape_125_Reshape_70359  );

  memcpy((sub_0030_arena + sub_0030_address_model_125_tf_reshape_125_Reshape_70359), buf_model_125_tf_reshape_125_Reshape_70359, 9408);
  // NPU Unit — sub_0030  (input @9408, output @9408)
  SCB_CleanDCache_by_Addr((void *)(sub_0030_arena + 9408U), 9408);
  if (sub_0030_invoke(clean_outputs) != 0) { if (_npu_status == 0) _npu_status = -30; }
  SCB_InvalidateDCache_by_Addr((void *)(sub_0030_arena + 9408U), 9408);

  // CPU Unit
  compute_sub_0031(compute_arena_sub_0031, buf_model_125_tf_compat_v1_transpose_313_transpose_70372, buf_model_125_tf_compat_v1_transpose_317_transpose_70376_70645  );

  memcpy((sub_0032_arena + sub_0032_address_model_125_tf_compat_v1_transpose_317_transpose_70376_70645), buf_model_125_tf_compat_v1_transpose_317_transpose_70376_70645, 9408);
  // NPU Unit — sub_0032  (input @9408, output @0 = final 1000-class logits)
  SCB_CleanDCache_by_Addr((void *)(sub_0032_arena + 9408U), 9408);
  if (sub_0032_invoke(clean_outputs) != 0) { if (_npu_status == 0) _npu_status = -32; }
  SCB_InvalidateDCache_by_Addr((void *)(sub_0032_arena + 0U), 1000);

  return _npu_status;
}
