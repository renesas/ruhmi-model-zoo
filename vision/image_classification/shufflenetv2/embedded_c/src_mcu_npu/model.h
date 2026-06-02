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

// NPU unit addresses
#include "sub_0000_tensors.h"
#include "sub_0002_tensors.h"
#include "sub_0004_tensors.h"
#include "sub_0006_tensors.h"
#include "sub_0008_tensors.h"
#include "sub_0010_tensors.h"
#include "sub_0012_tensors.h"
#include "sub_0014_tensors.h"
#include "sub_0016_tensors.h"
#include "sub_0018_tensors.h"
#include "sub_0020_tensors.h"
#include "sub_0022_tensors.h"
#include "sub_0024_tensors.h"
#include "sub_0026_tensors.h"
#include "sub_0028_tensors.h"
#include "sub_0030_tensors.h"
#include "sub_0032_tensors.h"

// Arenas for NPU units
extern uint8_t sub_0000_arena[kArenaSize_sub_0000];
extern uint8_t sub_0002_arena[kArenaSize_sub_0002];
extern uint8_t sub_0004_arena[kArenaSize_sub_0004];
extern uint8_t sub_0006_arena[kArenaSize_sub_0006];
extern uint8_t sub_0008_arena[kArenaSize_sub_0008];
extern uint8_t sub_0010_arena[kArenaSize_sub_0010];
extern uint8_t sub_0012_arena[kArenaSize_sub_0012];
extern uint8_t sub_0014_arena[kArenaSize_sub_0014];
extern uint8_t sub_0016_arena[kArenaSize_sub_0016];
extern uint8_t sub_0018_arena[kArenaSize_sub_0018];
extern uint8_t sub_0020_arena[kArenaSize_sub_0020];
extern uint8_t sub_0022_arena[kArenaSize_sub_0022];
extern uint8_t sub_0024_arena[kArenaSize_sub_0024];
extern uint8_t sub_0026_arena[kArenaSize_sub_0026];
extern uint8_t sub_0028_arena[kArenaSize_sub_0028];
extern uint8_t sub_0030_arena[kArenaSize_sub_0030];
extern uint8_t sub_0032_arena[kArenaSize_sub_0032];

// Buffers
extern int8_t buf_model_125_tf_compat_v1_transpose_193_transpose_70138[37632];
extern int8_t buf_model_125_tf_compat_v1_transpose_201_transpose_70154[37632];
extern int8_t buf_model_125_tf_compat_v1_transpose_209_transpose_70170[37632];
extern int8_t buf_model_125_tf_compat_v1_transpose_217_transpose_70186[37632];
extern int8_t buf_model_125_tf_compat_v1_transpose_225_transpose_70199[18816];
extern int8_t buf_model_125_tf_compat_v1_transpose_233_transpose_70215[18816];
extern int8_t buf_model_125_tf_compat_v1_transpose_241_transpose_70231[18816];
extern int8_t buf_model_125_tf_compat_v1_transpose_249_transpose_70247[18816];
extern int8_t buf_model_125_tf_compat_v1_transpose_257_transpose_70263[18816];
extern int8_t buf_model_125_tf_compat_v1_transpose_265_transpose_70279[18816];
extern int8_t buf_model_125_tf_compat_v1_transpose_273_transpose_70295[18816];
extern int8_t buf_model_125_tf_compat_v1_transpose_281_transpose_70311[18816];
extern int8_t buf_model_125_tf_compat_v1_transpose_289_transpose_70324[9408];
extern int8_t buf_model_125_tf_compat_v1_transpose_297_transpose_70340[9408];
extern int8_t buf_model_125_tf_compat_v1_transpose_305_transpose_70356[9408];
extern int8_t buf_model_125_tf_compat_v1_transpose_313_transpose_70372[9408];
extern int8_t buf_StatefulPartitionedCall_0_70379[1000];


int RunModel(bool clean_outputs);

  // Model input pointers
int8_t* GetModelInputPtr_serving_default_input_0();

  // Model output pointers
int8_t* GetModelOutputPtr_StatefulPartitionedCall_0_70379();

