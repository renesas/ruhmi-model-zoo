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
#include "sub_0034_tensors.h"
#include "sub_0036_tensors.h"
#include "sub_0038_tensors.h"
#include "sub_0040_tensors.h"

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
extern uint8_t sub_0034_arena[kArenaSize_sub_0034];
extern uint8_t sub_0036_arena[kArenaSize_sub_0036];
extern uint8_t sub_0038_arena[kArenaSize_sub_0038];
extern uint8_t sub_0040_arena[kArenaSize_sub_0040];

// Buffers
extern int8_t buf_conv2d_70116[147456];
extern int8_t buf_add_70120[147456];
extern int8_t buf_add_1_70124[147456];
extern int8_t buf_add_2_70130[73728];
extern int8_t buf_add_3_70134[73728];
extern int8_t buf_add_4_70138[73728];
extern int8_t buf_add_5_70144[36864];
extern int8_t buf_add_6_70148[36864];
extern int8_t buf_add_7_70152[36864];
extern int8_t buf_add_8_70158[18432];
extern int8_t buf_add_9_70162[18432];
extern int8_t buf_add_10_70166[18432];
extern int8_t buf_add_11_70171[4608];
extern int8_t buf_add_12_70175[4608];
extern int8_t buf_add_13_70179[4608];
extern int8_t buf_add_14_70184[1152];
extern int8_t buf_add_22_70195[1152];
extern int8_t buf_add_15_70188[1152];
extern int8_t buf_conv2d_28_70199[288];
extern int8_t buf_add_16_70196[1152];
extern int8_t buf_add_23_70207[288];
extern int8_t buf_conv2d_18_70200[288];
extern int8_t buf_add_17_70208[288];
extern int8_t buf_conv2d_30_70211[1];
extern int8_t buf_conv2d_20_70212[1404];


void RunModel(bool clean_outputs);

  // Model input pointers
int8_t* GetModelInputPtr_input_1();

  // Model output pointers
int8_t* GetModelOutputPtr_conv2d_30_70211();
int8_t* GetModelOutputPtr_conv2d_20_70212();

