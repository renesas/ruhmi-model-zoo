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

// Buffers for CPU units (moved to SDRAM to resolve RAM overflow)
__attribute__((aligned(16))) int8_t buf_serving_default_data_0[307200];
__attribute__((aligned(16))) int8_t buf_model_135_tf_compat_v1_transpose_246_transpose_70269_70636[92800];
__attribute__((aligned(16))) int8_t buf_model_135_tf_compat_v1_transpose_247_transpose_70267_70642[92800];
__attribute__((aligned(16))) int8_t buf_model_135_tf_compat_v1_transpose_254_transpose_70283_70654[92800];
__attribute__((aligned(16))) int8_t buf_model_135_tf_compat_v1_transpose_255_transpose_70281_70660[92800];
__attribute__((aligned(16))) int8_t buf_model_135_tf_compat_v1_transpose_262_transpose_70297_70672[92800];
__attribute__((aligned(16))) int8_t buf_model_135_tf_compat_v1_transpose_263_transpose_70295_70678[92800];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_270_transpose_70308_70690[185600];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_279_transpose_70329_70702[46400];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_280_transpose_70327_70708[46400];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_287_transpose_70343_70720[46400];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_288_transpose_70341_70726[46400];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_295_transpose_70357_70738[46400];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_296_transpose_70355_70744[46400];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_303_transpose_70371_70756[46400];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_304_transpose_70369_70762[46400];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_311_transpose_70385_70774[46400];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_312_transpose_70383_70780[46400];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_319_transpose_70399_70792[46400];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_320_transpose_70397_70798[46400];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_327_transpose_70413_70810[46400];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_328_transpose_70411_70816[46400];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_335_transpose_70424_70828[92800];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_344_transpose_70445_70840[23200];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_345_transpose_70443_70846[23200];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_352_transpose_70459_70858[23200];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_353_transpose_70457_70864[23200];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_360_transpose_70473_70876[23200];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_361_transpose_70471_70882[23200];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_compat_v1_transpose_368_transpose_70484_70894[46400];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_strided_slice_60_StridedSlice_70620[128000];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_strided_slice_61_StridedSlice_70623[51200];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_strided_slice_62_StridedSlice_70604[32000];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_strided_slice_63_StridedSlice_70607[12800];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_strided_slice_64_StridedSlice_70588[8000];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_strided_slice_65_StridedSlice_70591[3200];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_strided_slice_66_StridedSlice_70572[2000];
__attribute__((aligned(16), section(".sdram"))) int8_t buf_model_135_tf_strided_slice_67_StridedSlice_70575[800];

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
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0033[kBufferSize_sub_0033];

  // Model input pointers
int8_t* GetModelInputPtr_serving_default_data_0() {
  return (int8_t*) (sub_0000_arena + sub_0000_address_serving_default_data_0);
}


  // Model output pointers
int8_t* GetModelOutputPtr_StatefulPartitionedCall_0_70628() {
  return (int8_t*) (sub_0034_arena + sub_0034_address_StatefulPartitionedCall_0_70628);
}


void RunModel(bool clean_outputs) {
  // Buffers for NPU units
  int8_t* buf_model_135_tf_compat_v1_transpose_242_transpose_70262 = (int8_t*) (sub_0000_arena + sub_0000_address_model_135_tf_compat_v1_transpose_242_transpose_70262);
  int8_t* buf_model_135_tf_compat_v1_transpose_250_transpose_70276 = (int8_t*) (sub_0002_arena + sub_0002_address_model_135_tf_compat_v1_transpose_250_transpose_70276);
  int8_t* buf_model_135_tf_compat_v1_transpose_258_transpose_70290 = (int8_t*) (sub_0004_arena + sub_0004_address_model_135_tf_compat_v1_transpose_258_transpose_70290);
  int8_t* buf_model_135_tf_compat_v1_transpose_266_transpose_70304 = (int8_t*) (sub_0006_arena + sub_0006_address_model_135_tf_compat_v1_transpose_266_transpose_70304);
  int8_t* buf_model_135_tf_compat_v1_transpose_275_transpose_70320 = (int8_t*) (sub_0008_arena + sub_0008_address_model_135_tf_compat_v1_transpose_275_transpose_70320);
  int8_t* buf_model_135_tf_nn_leaky_relu_82_LeakyRelu_70325 = (int8_t*) (sub_0008_arena + sub_0008_address_model_135_tf_nn_leaky_relu_82_LeakyRelu_70325);
  int8_t* buf_model_135_tf_compat_v1_transpose_283_transpose_70336 = (int8_t*) (sub_0010_arena + sub_0010_address_model_135_tf_compat_v1_transpose_283_transpose_70336);
  int8_t* buf_model_135_tf_compat_v1_transpose_291_transpose_70350 = (int8_t*) (sub_0012_arena + sub_0012_address_model_135_tf_compat_v1_transpose_291_transpose_70350);
  int8_t* buf_model_135_tf_compat_v1_transpose_299_transpose_70364 = (int8_t*) (sub_0014_arena + sub_0014_address_model_135_tf_compat_v1_transpose_299_transpose_70364);
  int8_t* buf_model_135_tf_compat_v1_transpose_307_transpose_70378 = (int8_t*) (sub_0016_arena + sub_0016_address_model_135_tf_compat_v1_transpose_307_transpose_70378);
  int8_t* buf_model_135_tf_compat_v1_transpose_315_transpose_70392 = (int8_t*) (sub_0018_arena + sub_0018_address_model_135_tf_compat_v1_transpose_315_transpose_70392);
  int8_t* buf_model_135_tf_compat_v1_transpose_323_transpose_70406 = (int8_t*) (sub_0020_arena + sub_0020_address_model_135_tf_compat_v1_transpose_323_transpose_70406);
  int8_t* buf_model_135_tf_compat_v1_transpose_331_transpose_70420 = (int8_t*) (sub_0022_arena + sub_0022_address_model_135_tf_compat_v1_transpose_331_transpose_70420);
  int8_t* buf_model_135_tf_compat_v1_transpose_340_transpose_70436 = (int8_t*) (sub_0024_arena + sub_0024_address_model_135_tf_compat_v1_transpose_340_transpose_70436);
  int8_t* buf_model_135_tf_nn_leaky_relu_100_LeakyRelu_70441 = (int8_t*) (sub_0024_arena + sub_0024_address_model_135_tf_nn_leaky_relu_100_LeakyRelu_70441);
  int8_t* buf_model_135_tf_compat_v1_transpose_348_transpose_70452 = (int8_t*) (sub_0026_arena + sub_0026_address_model_135_tf_compat_v1_transpose_348_transpose_70452);
  int8_t* buf_model_135_tf_compat_v1_transpose_356_transpose_70466 = (int8_t*) (sub_0028_arena + sub_0028_address_model_135_tf_compat_v1_transpose_356_transpose_70466);
  int8_t* buf_model_135_tf_compat_v1_transpose_364_transpose_70480 = (int8_t*) (sub_0030_arena + sub_0030_address_model_135_tf_compat_v1_transpose_364_transpose_70480);
  int8_t* buf_model_135_tf_math_add_211_Add_model_135_tf_nn_convolution_133_convolution_Const_70619 = (int8_t*) (sub_0032_arena + sub_0032_address_model_135_tf_math_add_211_Add_model_135_tf_nn_convolution_133_convolution_Const_70619);
  int8_t* buf_model_135_tf_math_add_226_Add_model_135_tf_nn_convolution_142_convolution_Const_1_70603 = (int8_t*) (sub_0032_arena + sub_0032_address_model_135_tf_math_add_226_Add_model_135_tf_nn_convolution_142_convolution_Const_1_70603);
  int8_t* buf_model_135_tf_math_add_241_Add_model_135_tf_nn_convolution_151_convolution_Const_2_70587 = (int8_t*) (sub_0032_arena + sub_0032_address_model_135_tf_math_add_241_Add_model_135_tf_nn_convolution_151_convolution_Const_2_70587);
  int8_t* buf_model_135_tf_math_add_244_Add_model_135_tf_nn_convolution_153_convolution_Const_3_70571 = (int8_t*) (sub_0032_arena + sub_0032_address_model_135_tf_math_add_244_Add_model_135_tf_nn_convolution_153_convolution_Const_3_70571);
  int8_t* buf_StatefulPartitionedCall_0_70628 = (int8_t*) (sub_0034_arena + sub_0034_address_StatefulPartitionedCall_0_70628);

  // NPU Unit
  sub_0000_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0001(compute_arena_sub_0001, buf_model_135_tf_compat_v1_transpose_242_transpose_70262, buf_model_135_tf_compat_v1_transpose_246_transpose_70269_70636, buf_model_135_tf_compat_v1_transpose_247_transpose_70267_70642  );

  memcpy((sub_0002_arena + sub_0002_address_model_135_tf_compat_v1_transpose_246_transpose_70269_70636), buf_model_135_tf_compat_v1_transpose_246_transpose_70269_70636, 92800);
  memcpy((sub_0002_arena + sub_0002_address_model_135_tf_compat_v1_transpose_247_transpose_70267_70642), buf_model_135_tf_compat_v1_transpose_247_transpose_70267_70642, 92800);
  // NPU Unit
  sub_0002_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0003(compute_arena_sub_0003, buf_model_135_tf_compat_v1_transpose_250_transpose_70276, buf_model_135_tf_compat_v1_transpose_254_transpose_70283_70654, buf_model_135_tf_compat_v1_transpose_255_transpose_70281_70660  );

  memcpy((sub_0004_arena + sub_0004_address_model_135_tf_compat_v1_transpose_254_transpose_70283_70654), buf_model_135_tf_compat_v1_transpose_254_transpose_70283_70654, 92800);
  memcpy((sub_0004_arena + sub_0004_address_model_135_tf_compat_v1_transpose_255_transpose_70281_70660), buf_model_135_tf_compat_v1_transpose_255_transpose_70281_70660, 92800);
  // NPU Unit
  sub_0004_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0005(compute_arena_sub_0005, buf_model_135_tf_compat_v1_transpose_258_transpose_70290, buf_model_135_tf_compat_v1_transpose_262_transpose_70297_70672, buf_model_135_tf_compat_v1_transpose_263_transpose_70295_70678  );

  memcpy((sub_0006_arena + sub_0006_address_model_135_tf_compat_v1_transpose_262_transpose_70297_70672), buf_model_135_tf_compat_v1_transpose_262_transpose_70297_70672, 92800);
  memcpy((sub_0006_arena + sub_0006_address_model_135_tf_compat_v1_transpose_263_transpose_70295_70678), buf_model_135_tf_compat_v1_transpose_263_transpose_70295_70678, 92800);
  // NPU Unit
  sub_0006_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0007(compute_arena_sub_0007, buf_model_135_tf_compat_v1_transpose_266_transpose_70304, buf_model_135_tf_compat_v1_transpose_270_transpose_70308_70690  );

  memcpy((sub_0008_arena + sub_0008_address_model_135_tf_compat_v1_transpose_270_transpose_70308_70690), buf_model_135_tf_compat_v1_transpose_270_transpose_70308_70690, 185600);
  // NPU Unit
  sub_0008_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0009(compute_arena_sub_0009, buf_model_135_tf_compat_v1_transpose_275_transpose_70320, buf_model_135_tf_compat_v1_transpose_279_transpose_70329_70702, buf_model_135_tf_compat_v1_transpose_280_transpose_70327_70708  );

  memcpy((sub_0010_arena + sub_0010_address_model_135_tf_compat_v1_transpose_279_transpose_70329_70702), buf_model_135_tf_compat_v1_transpose_279_transpose_70329_70702, 46400);
  memcpy((sub_0010_arena + sub_0010_address_model_135_tf_compat_v1_transpose_280_transpose_70327_70708), buf_model_135_tf_compat_v1_transpose_280_transpose_70327_70708, 46400);
  // NPU Unit
  sub_0010_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0011(compute_arena_sub_0011, buf_model_135_tf_compat_v1_transpose_283_transpose_70336, buf_model_135_tf_compat_v1_transpose_287_transpose_70343_70720, buf_model_135_tf_compat_v1_transpose_288_transpose_70341_70726  );

  memcpy((sub_0012_arena + sub_0012_address_model_135_tf_compat_v1_transpose_287_transpose_70343_70720), buf_model_135_tf_compat_v1_transpose_287_transpose_70343_70720, 46400);
  memcpy((sub_0012_arena + sub_0012_address_model_135_tf_compat_v1_transpose_288_transpose_70341_70726), buf_model_135_tf_compat_v1_transpose_288_transpose_70341_70726, 46400);
  // NPU Unit
  sub_0012_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0013(compute_arena_sub_0013, buf_model_135_tf_compat_v1_transpose_291_transpose_70350, buf_model_135_tf_compat_v1_transpose_295_transpose_70357_70738, buf_model_135_tf_compat_v1_transpose_296_transpose_70355_70744  );

  memcpy((sub_0014_arena + sub_0014_address_model_135_tf_compat_v1_transpose_295_transpose_70357_70738), buf_model_135_tf_compat_v1_transpose_295_transpose_70357_70738, 46400);
  memcpy((sub_0014_arena + sub_0014_address_model_135_tf_compat_v1_transpose_296_transpose_70355_70744), buf_model_135_tf_compat_v1_transpose_296_transpose_70355_70744, 46400);
  // NPU Unit
  sub_0014_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0015(compute_arena_sub_0015, buf_model_135_tf_compat_v1_transpose_299_transpose_70364, buf_model_135_tf_compat_v1_transpose_303_transpose_70371_70756, buf_model_135_tf_compat_v1_transpose_304_transpose_70369_70762  );

  memcpy((sub_0016_arena + sub_0016_address_model_135_tf_compat_v1_transpose_303_transpose_70371_70756), buf_model_135_tf_compat_v1_transpose_303_transpose_70371_70756, 46400);
  memcpy((sub_0016_arena + sub_0016_address_model_135_tf_compat_v1_transpose_304_transpose_70369_70762), buf_model_135_tf_compat_v1_transpose_304_transpose_70369_70762, 46400);
  // NPU Unit
  sub_0016_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0017(compute_arena_sub_0017, buf_model_135_tf_compat_v1_transpose_307_transpose_70378, buf_model_135_tf_compat_v1_transpose_311_transpose_70385_70774, buf_model_135_tf_compat_v1_transpose_312_transpose_70383_70780  );

  memcpy((sub_0018_arena + sub_0018_address_model_135_tf_compat_v1_transpose_311_transpose_70385_70774), buf_model_135_tf_compat_v1_transpose_311_transpose_70385_70774, 46400);
  memcpy((sub_0018_arena + sub_0018_address_model_135_tf_compat_v1_transpose_312_transpose_70383_70780), buf_model_135_tf_compat_v1_transpose_312_transpose_70383_70780, 46400);
  // NPU Unit
  sub_0018_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0019(compute_arena_sub_0019, buf_model_135_tf_compat_v1_transpose_315_transpose_70392, buf_model_135_tf_compat_v1_transpose_319_transpose_70399_70792, buf_model_135_tf_compat_v1_transpose_320_transpose_70397_70798  );

  memcpy((sub_0020_arena + sub_0020_address_model_135_tf_compat_v1_transpose_319_transpose_70399_70792), buf_model_135_tf_compat_v1_transpose_319_transpose_70399_70792, 46400);
  memcpy((sub_0020_arena + sub_0020_address_model_135_tf_compat_v1_transpose_320_transpose_70397_70798), buf_model_135_tf_compat_v1_transpose_320_transpose_70397_70798, 46400);
  // NPU Unit
  sub_0020_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0021(compute_arena_sub_0021, buf_model_135_tf_compat_v1_transpose_323_transpose_70406, buf_model_135_tf_compat_v1_transpose_327_transpose_70413_70810, buf_model_135_tf_compat_v1_transpose_328_transpose_70411_70816  );

  memcpy((sub_0022_arena + sub_0022_address_model_135_tf_compat_v1_transpose_327_transpose_70413_70810), buf_model_135_tf_compat_v1_transpose_327_transpose_70413_70810, 46400);
  memcpy((sub_0022_arena + sub_0022_address_model_135_tf_compat_v1_transpose_328_transpose_70411_70816), buf_model_135_tf_compat_v1_transpose_328_transpose_70411_70816, 46400);
  // NPU Unit
  sub_0022_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0023(compute_arena_sub_0023, buf_model_135_tf_compat_v1_transpose_331_transpose_70420, buf_model_135_tf_compat_v1_transpose_335_transpose_70424_70828  );

  memcpy((sub_0024_arena + sub_0024_address_model_135_tf_compat_v1_transpose_335_transpose_70424_70828), buf_model_135_tf_compat_v1_transpose_335_transpose_70424_70828, 92800);
  // NPU Unit
  sub_0024_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0025(compute_arena_sub_0025, buf_model_135_tf_compat_v1_transpose_340_transpose_70436, buf_model_135_tf_compat_v1_transpose_344_transpose_70445_70840, buf_model_135_tf_compat_v1_transpose_345_transpose_70443_70846  );

  memcpy((sub_0026_arena + sub_0026_address_model_135_tf_compat_v1_transpose_344_transpose_70445_70840), buf_model_135_tf_compat_v1_transpose_344_transpose_70445_70840, 23200);
  memcpy((sub_0026_arena + sub_0026_address_model_135_tf_compat_v1_transpose_345_transpose_70443_70846), buf_model_135_tf_compat_v1_transpose_345_transpose_70443_70846, 23200);
  // NPU Unit
  sub_0026_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0027(compute_arena_sub_0027, buf_model_135_tf_compat_v1_transpose_348_transpose_70452, buf_model_135_tf_compat_v1_transpose_352_transpose_70459_70858, buf_model_135_tf_compat_v1_transpose_353_transpose_70457_70864  );

  memcpy((sub_0028_arena + sub_0028_address_model_135_tf_compat_v1_transpose_352_transpose_70459_70858), buf_model_135_tf_compat_v1_transpose_352_transpose_70459_70858, 23200);
  memcpy((sub_0028_arena + sub_0028_address_model_135_tf_compat_v1_transpose_353_transpose_70457_70864), buf_model_135_tf_compat_v1_transpose_353_transpose_70457_70864, 23200);
  // NPU Unit
  sub_0028_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0029(compute_arena_sub_0029, buf_model_135_tf_compat_v1_transpose_356_transpose_70466, buf_model_135_tf_compat_v1_transpose_360_transpose_70473_70876, buf_model_135_tf_compat_v1_transpose_361_transpose_70471_70882  );

  memcpy((sub_0030_arena + sub_0030_address_model_135_tf_compat_v1_transpose_360_transpose_70473_70876), buf_model_135_tf_compat_v1_transpose_360_transpose_70473_70876, 23200);
  memcpy((sub_0030_arena + sub_0030_address_model_135_tf_compat_v1_transpose_361_transpose_70471_70882), buf_model_135_tf_compat_v1_transpose_361_transpose_70471_70882, 23200);
  // NPU Unit
  sub_0030_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0031(compute_arena_sub_0031, buf_model_135_tf_compat_v1_transpose_364_transpose_70480, buf_model_135_tf_compat_v1_transpose_368_transpose_70484_70894  );

  memcpy((sub_0032_arena + sub_0032_address_model_135_tf_compat_v1_transpose_368_transpose_70484_70894), buf_model_135_tf_compat_v1_transpose_368_transpose_70484_70894, 46400);
  memcpy((sub_0032_arena + sub_0032_address_model_135_tf_nn_leaky_relu_100_LeakyRelu_70441), buf_model_135_tf_nn_leaky_relu_100_LeakyRelu_70441, 38400);
  memcpy((sub_0032_arena + sub_0032_address_model_135_tf_nn_leaky_relu_82_LeakyRelu_70325), buf_model_135_tf_nn_leaky_relu_82_LeakyRelu_70325, 153600);
  // NPU Unit
  sub_0032_invoke(clean_outputs);

  // CPU Unit
  compute_sub_0033(compute_arena_sub_0033, buf_model_135_tf_math_add_211_Add_model_135_tf_nn_convolution_133_convolution_Const_70619, buf_model_135_tf_math_add_226_Add_model_135_tf_nn_convolution_142_convolution_Const_1_70603, buf_model_135_tf_math_add_241_Add_model_135_tf_nn_convolution_151_convolution_Const_2_70587, buf_model_135_tf_math_add_244_Add_model_135_tf_nn_convolution_153_convolution_Const_3_70571, buf_model_135_tf_strided_slice_60_StridedSlice_70620, buf_model_135_tf_strided_slice_61_StridedSlice_70623, buf_model_135_tf_strided_slice_62_StridedSlice_70604, buf_model_135_tf_strided_slice_63_StridedSlice_70607, buf_model_135_tf_strided_slice_64_StridedSlice_70588, buf_model_135_tf_strided_slice_65_StridedSlice_70591, buf_model_135_tf_strided_slice_66_StridedSlice_70572, buf_model_135_tf_strided_slice_67_StridedSlice_70575  );

  memcpy((sub_0034_arena + sub_0034_address_model_135_tf_strided_slice_60_StridedSlice_70620), buf_model_135_tf_strided_slice_60_StridedSlice_70620, 128000);
  memcpy((sub_0034_arena + sub_0034_address_model_135_tf_strided_slice_61_StridedSlice_70623), buf_model_135_tf_strided_slice_61_StridedSlice_70623, 51200);
  memcpy((sub_0034_arena + sub_0034_address_model_135_tf_strided_slice_62_StridedSlice_70604), buf_model_135_tf_strided_slice_62_StridedSlice_70604, 32000);
  memcpy((sub_0034_arena + sub_0034_address_model_135_tf_strided_slice_63_StridedSlice_70607), buf_model_135_tf_strided_slice_63_StridedSlice_70607, 12800);
  memcpy((sub_0034_arena + sub_0034_address_model_135_tf_strided_slice_64_StridedSlice_70588), buf_model_135_tf_strided_slice_64_StridedSlice_70588, 8000);
  memcpy((sub_0034_arena + sub_0034_address_model_135_tf_strided_slice_65_StridedSlice_70591), buf_model_135_tf_strided_slice_65_StridedSlice_70591, 3200);
  memcpy((sub_0034_arena + sub_0034_address_model_135_tf_strided_slice_66_StridedSlice_70572), buf_model_135_tf_strided_slice_66_StridedSlice_70572, 2000);
  memcpy((sub_0034_arena + sub_0034_address_model_135_tf_strided_slice_67_StridedSlice_70575), buf_model_135_tf_strided_slice_67_StridedSlice_70575, 800);
  // NPU Unit
  sub_0034_invoke(clean_outputs);

}
