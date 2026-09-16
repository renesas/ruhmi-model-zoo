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
#include "compute_sub_0033.h"
#include "sub_0034_invoke.h"
#include "compute_sub_0035.h"
#include "sub_0036_invoke.h"
#include "compute_sub_0037.h"
#include "sub_0038_invoke.h"
#include "compute_sub_0039.h"
#include "sub_0040_invoke.h"

// Buffers for CPU units
int8_t buf_input_1[110592];
int8_t buf_p_re_lu_70117[147456];
int8_t buf_p_re_lu_1_70121[147456];
int8_t buf_p_re_lu_2_70125[147456];
int8_t buf_p_re_lu_3_70131[73728];
int8_t buf_p_re_lu_4_70135[73728];
int8_t buf_p_re_lu_5_70139[73728];
int8_t buf_p_re_lu_6_70145[36864];
int8_t buf_p_re_lu_7_70149[36864];
int8_t buf_p_re_lu_8_70153[36864];
int8_t buf_p_re_lu_9_70159[18432];
int8_t buf_p_re_lu_10_70163[18432];
int8_t buf_p_re_lu_11_70167[18432];
int8_t buf_p_re_lu_12_70172[4608];
int8_t buf_p_re_lu_13_70176[4608];
int8_t buf_p_re_lu_14_70180[4608];
int8_t buf_p_re_lu_15_70185[1152];
int8_t buf_p_re_lu_25_70197[1152];
int8_t buf_p_re_lu_16_70189[1152];
int8_t buf_p_re_lu_26_70201[288];
int8_t buf_p_re_lu_17_70198[1152];
int8_t buf_p_re_lu_27_70209[288];
int8_t buf_p_re_lu_18_70202[288];
int8_t buf_p_re_lu_19_70210[288];

// Arenas for CPU units
uint8_t compute_arena_sub_0001[kBufferSize_sub_0001];
uint8_t compute_arena_sub_0003[kBufferSize_sub_0003];
uint8_t compute_arena_sub_0005[kBufferSize_sub_0005];
uint8_t compute_arena_sub_0007[kBufferSize_sub_0007];
uint8_t compute_arena_sub_0009[kBufferSize_sub_0009];
uint8_t compute_arena_sub_0011[kBufferSize_sub_0011];
uint8_t compute_arena_sub_0013[kBufferSize_sub_0013];
uint8_t compute_arena_sub_0015[kBufferSize_sub_0015];
uint8_t compute_arena_sub_0017[kBufferSize_sub_0017];
uint8_t compute_arena_sub_0019[kBufferSize_sub_0019];
uint8_t compute_arena_sub_0021[kBufferSize_sub_0021];
uint8_t compute_arena_sub_0023[kBufferSize_sub_0023];
uint8_t compute_arena_sub_0025[kBufferSize_sub_0025];
uint8_t compute_arena_sub_0027[kBufferSize_sub_0027];
uint8_t compute_arena_sub_0029[kBufferSize_sub_0029];
uint8_t compute_arena_sub_0031[kBufferSize_sub_0031];
uint8_t compute_arena_sub_0033[kBufferSize_sub_0033];
uint8_t compute_arena_sub_0035[kBufferSize_sub_0035];
uint8_t compute_arena_sub_0037[kBufferSize_sub_0037];
uint8_t compute_arena_sub_0039[kBufferSize_sub_0039];

  // Model input pointers
int8_t* GetModelInputPtr_input_1() {
  return (int8_t*) (sub_0000_arena + sub_0000_address_input_1);
}


  // Model output pointers
int8_t* GetModelOutputPtr_conv2d_30_70211() {
  return (int8_t*) (sub_0036_arena + sub_0036_address_conv2d_30_70211);
}

int8_t* GetModelOutputPtr_conv2d_20_70212() {
  return (int8_t*) (sub_0040_arena + sub_0040_address_conv2d_20_70212);
}


void RunModel(bool clean_outputs) {
  // Buffers for NPU units
  int8_t* buf_conv2d_70116 = (int8_t*) (sub_0000_arena + sub_0000_address_conv2d_70116);
  int8_t* buf_add_70120 = (int8_t*) (sub_0002_arena + sub_0002_address_add_70120);
  int8_t* buf_add_1_70124 = (int8_t*) (sub_0004_arena + sub_0004_address_add_1_70124);
  int8_t* buf_add_2_70130 = (int8_t*) (sub_0006_arena + sub_0006_address_add_2_70130);
  int8_t* buf_add_3_70134 = (int8_t*) (sub_0008_arena + sub_0008_address_add_3_70134);
  int8_t* buf_add_4_70138 = (int8_t*) (sub_0010_arena + sub_0010_address_add_4_70138);
  int8_t* buf_add_5_70144 = (int8_t*) (sub_0012_arena + sub_0012_address_add_5_70144);
  int8_t* buf_add_6_70148 = (int8_t*) (sub_0014_arena + sub_0014_address_add_6_70148);
  int8_t* buf_add_7_70152 = (int8_t*) (sub_0016_arena + sub_0016_address_add_7_70152);
  int8_t* buf_add_8_70158 = (int8_t*) (sub_0018_arena + sub_0018_address_add_8_70158);
  int8_t* buf_add_9_70162 = (int8_t*) (sub_0020_arena + sub_0020_address_add_9_70162);
  int8_t* buf_add_10_70166 = (int8_t*) (sub_0022_arena + sub_0022_address_add_10_70166);
  int8_t* buf_add_11_70171 = (int8_t*) (sub_0024_arena + sub_0024_address_add_11_70171);
  int8_t* buf_add_12_70175 = (int8_t*) (sub_0026_arena + sub_0026_address_add_12_70175);
  int8_t* buf_add_13_70179 = (int8_t*) (sub_0028_arena + sub_0028_address_add_13_70179);
  int8_t* buf_add_14_70184 = (int8_t*) (sub_0030_arena + sub_0030_address_add_14_70184);
  int8_t* buf_add_22_70195 = (int8_t*) (sub_0030_arena + sub_0030_address_add_22_70195);
  int8_t* buf_add_15_70188 = (int8_t*) (sub_0032_arena + sub_0032_address_add_15_70188);
  int8_t* buf_conv2d_28_70199 = (int8_t*) (sub_0032_arena + sub_0032_address_conv2d_28_70199);
  int8_t* buf_add_16_70196 = (int8_t*) (sub_0034_arena + sub_0034_address_add_16_70196);
  int8_t* buf_add_23_70207 = (int8_t*) (sub_0034_arena + sub_0034_address_add_23_70207);
  int8_t* buf_conv2d_18_70200 = (int8_t*) (sub_0036_arena + sub_0036_address_conv2d_18_70200);
  int8_t* buf_add_17_70208 = (int8_t*) (sub_0038_arena + sub_0038_address_add_17_70208);
  int8_t* buf_conv2d_30_70211 = (int8_t*) (sub_0036_arena + sub_0036_address_conv2d_30_70211);
  int8_t* buf_conv2d_20_70212 = (int8_t*) (sub_0040_arena + sub_0040_address_conv2d_20_70212);

  // NPU Unit
  sub_0000_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0001(compute_arena_sub_0001, buf_conv2d_70116, buf_p_re_lu_70117  );

  memcpy((sub_0002_arena + sub_0002_address_p_re_lu_70117), buf_p_re_lu_70117, 147456);
  // NPU Unit
  sub_0002_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0003(compute_arena_sub_0003, buf_add_70120, buf_p_re_lu_1_70121  );

  memcpy((sub_0004_arena + sub_0004_address_p_re_lu_1_70121), buf_p_re_lu_1_70121, 147456);
  // NPU Unit
  sub_0004_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0005(compute_arena_sub_0005, buf_add_1_70124, buf_p_re_lu_2_70125  );

  memcpy((sub_0006_arena + sub_0006_address_p_re_lu_2_70125), buf_p_re_lu_2_70125, 147456);
  // NPU Unit
  sub_0006_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0007(compute_arena_sub_0007, buf_add_2_70130, buf_p_re_lu_3_70131  );

  memcpy((sub_0008_arena + sub_0008_address_p_re_lu_3_70131), buf_p_re_lu_3_70131, 73728);
  // NPU Unit
  sub_0008_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0009(compute_arena_sub_0009, buf_add_3_70134, buf_p_re_lu_4_70135  );

  memcpy((sub_0010_arena + sub_0010_address_p_re_lu_4_70135), buf_p_re_lu_4_70135, 73728);
  // NPU Unit
  sub_0010_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0011(compute_arena_sub_0011, buf_add_4_70138, buf_p_re_lu_5_70139  );

  memcpy((sub_0012_arena + sub_0012_address_p_re_lu_5_70139), buf_p_re_lu_5_70139, 73728);
  // NPU Unit
  sub_0012_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0013(compute_arena_sub_0013, buf_add_5_70144, buf_p_re_lu_6_70145  );

  memcpy((sub_0014_arena + sub_0014_address_p_re_lu_6_70145), buf_p_re_lu_6_70145, 36864);
  // NPU Unit
  sub_0014_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0015(compute_arena_sub_0015, buf_add_6_70148, buf_p_re_lu_7_70149  );

  memcpy((sub_0016_arena + sub_0016_address_p_re_lu_7_70149), buf_p_re_lu_7_70149, 36864);
  // NPU Unit
  sub_0016_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0017(compute_arena_sub_0017, buf_add_7_70152, buf_p_re_lu_8_70153  );

  memcpy((sub_0018_arena + sub_0018_address_p_re_lu_8_70153), buf_p_re_lu_8_70153, 36864);
  // NPU Unit
  sub_0018_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0019(compute_arena_sub_0019, buf_add_8_70158, buf_p_re_lu_9_70159  );

  memcpy((sub_0020_arena + sub_0020_address_p_re_lu_9_70159), buf_p_re_lu_9_70159, 18432);
  // NPU Unit
  sub_0020_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0021(compute_arena_sub_0021, buf_add_9_70162, buf_p_re_lu_10_70163  );

  memcpy((sub_0022_arena + sub_0022_address_p_re_lu_10_70163), buf_p_re_lu_10_70163, 18432);
  // NPU Unit
  sub_0022_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0023(compute_arena_sub_0023, buf_add_10_70166, buf_p_re_lu_11_70167  );

  memcpy((sub_0024_arena + sub_0024_address_p_re_lu_11_70167), buf_p_re_lu_11_70167, 18432);
  // NPU Unit
  sub_0024_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0025(compute_arena_sub_0025, buf_add_11_70171, buf_p_re_lu_12_70172  );

  memcpy((sub_0026_arena + sub_0026_address_p_re_lu_12_70172), buf_p_re_lu_12_70172, 4608);
  // NPU Unit
  sub_0026_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0027(compute_arena_sub_0027, buf_add_12_70175, buf_p_re_lu_13_70176  );

  memcpy((sub_0028_arena + sub_0028_address_p_re_lu_13_70176), buf_p_re_lu_13_70176, 4608);
  // NPU Unit
  sub_0028_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0029(compute_arena_sub_0029, buf_add_13_70179, buf_p_re_lu_14_70180  );

  memcpy((sub_0030_arena + sub_0030_address_p_re_lu_14_70180), buf_p_re_lu_14_70180, 4608);
  // NPU Unit
  sub_0030_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0031(compute_arena_sub_0031, buf_add_14_70184, buf_add_22_70195, buf_p_re_lu_15_70185, buf_p_re_lu_25_70197  );

  memcpy((sub_0032_arena + sub_0032_address_p_re_lu_15_70185), buf_p_re_lu_15_70185, 1152);
  memcpy((sub_0032_arena + sub_0032_address_p_re_lu_25_70197), buf_p_re_lu_25_70197, 1152);
  // NPU Unit
  sub_0032_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0033(compute_arena_sub_0033, buf_add_15_70188, buf_conv2d_28_70199, buf_p_re_lu_16_70189, buf_p_re_lu_26_70201  );

  memcpy((sub_0034_arena + sub_0034_address_p_re_lu_16_70189), buf_p_re_lu_16_70189, 1152);
  memcpy((sub_0034_arena + sub_0034_address_p_re_lu_26_70201), buf_p_re_lu_26_70201, 288);
  // NPU Unit
  sub_0034_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0035(compute_arena_sub_0035, buf_add_16_70196, buf_add_23_70207, buf_p_re_lu_17_70198, buf_p_re_lu_27_70209  );

  memcpy((sub_0036_arena + sub_0036_address_p_re_lu_17_70198), buf_p_re_lu_17_70198, 1152);
  memcpy((sub_0036_arena + sub_0036_address_p_re_lu_27_70209), buf_p_re_lu_27_70209, 288);
  // NPU Unit
  sub_0036_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0037(compute_arena_sub_0037, buf_conv2d_18_70200, buf_p_re_lu_18_70202  );

  memcpy((sub_0038_arena + sub_0038_address_p_re_lu_18_70202), buf_p_re_lu_18_70202, 288);
  // NPU Unit
  sub_0038_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0039(compute_arena_sub_0039, buf_add_17_70208, buf_p_re_lu_19_70210  );

  memcpy((sub_0040_arena + sub_0040_address_p_re_lu_19_70210), buf_p_re_lu_19_70210, 288);
  // NPU Unit
  sub_0040_invoke(clean_outputs);

}
