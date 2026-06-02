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

#include "model.h"

// CPU compute declarations
#include "compute_sub_0000.h"
#include "compute_sub_0001.h"
#include "sub_0002_invoke.h"

// Buffers for CPU units
int8_t buf_serving_default_images_0[150528];
int8_t buf_model_17_tf_strided_slice_8_StridedSlice_70185[37632];
int8_t buf_model_17_tf_strided_slice_9_StridedSlice_70186[37632];
int8_t buf_model_17_tf_strided_slice_10_StridedSlice_70183[37632];
int8_t buf_model_17_tf_strided_slice_11_StridedSlice_70184[37632];
int8_t buf_PartitionedCall_0_70478[87465];

// Arenas for CPU units
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0000[kBufferSize_sub_0000];
__attribute__((aligned(16), section(".sdram"))) uint8_t compute_arena_sub_0001[kBufferSize_sub_0001];

  // Model input pointers
int8_t* GetModelInputPtr_serving_default_images_0() {
  return buf_serving_default_images_0;
}


  // Model output pointers
int8_t* GetModelOutputPtr_PartitionedCall_0_70478() {
  return (int8_t*) (sub_0002_arena + sub_0002_address_PartitionedCall_0_70478);
}


void RunModel(bool clean_outputs) {
// CPU Unit
  compute_sub_0000(compute_arena_sub_0000, buf_serving_default_images_0, buf_model_17_tf_strided_slice_8_StridedSlice_70185, buf_model_17_tf_strided_slice_9_StridedSlice_70186  );

// CPU Unit
  compute_sub_0001(compute_arena_sub_0001, buf_serving_default_images_0, buf_model_17_tf_strided_slice_10_StridedSlice_70183, buf_model_17_tf_strided_slice_11_StridedSlice_70184  );

  // Feed CPU outputs to NPU input tensors in the shared arena.
  (void)memcpy((void *) (sub_0002_arena + sub_0002_address_model_17_tf_strided_slice_8_StridedSlice_70185),
               (const void *) buf_model_17_tf_strided_slice_8_StridedSlice_70185,
               sizeof(buf_model_17_tf_strided_slice_8_StridedSlice_70185));
  (void)memcpy((void *) (sub_0002_arena + sub_0002_address_model_17_tf_strided_slice_10_StridedSlice_70183),
               (const void *) buf_model_17_tf_strided_slice_10_StridedSlice_70183,
               sizeof(buf_model_17_tf_strided_slice_10_StridedSlice_70183));
  (void)memcpy((void *) (sub_0002_arena + sub_0002_address_model_17_tf_strided_slice_9_StridedSlice_70186),
               (const void *) buf_model_17_tf_strided_slice_9_StridedSlice_70186,
               sizeof(buf_model_17_tf_strided_slice_9_StridedSlice_70186));
  (void)memcpy((void *) (sub_0002_arena + sub_0002_address_model_17_tf_strided_slice_11_StridedSlice_70184),
               (const void *) buf_model_17_tf_strided_slice_11_StridedSlice_70184,
               sizeof(buf_model_17_tf_strided_slice_11_StridedSlice_70184));

// NPU Unit
  sub_0002_invoke(clean_outputs);

}
