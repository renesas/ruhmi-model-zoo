/***********************************************************************************************************************
 * Copyright [2020-2024] Renesas Electronics Corporation and/or its affiliates. All Rights Reserved.
 *
 * This software and documentation are supplied by Renesas Electronics Corporation and/or its affiliates and may only
 * be used with products of Renesas Electronics Corp. and its affiliates ("Renesas"). No other uses are authorized.
 * Renesas products are sold pursuant to Renesas terms and conditions of sale. Purchasers are solely responsible for
 * the selection and use of Renesas products and Renesas assumes no liability. No license, express or implied, to any
 * intellectual property right is granted by Renesas. This software is protected under all applicable laws, including
 * copyright laws. Renesas reserves the right to change or discontinue this software and/or this documentation.
 * THE SOFTWARE AND DOCUMENTATION IS DELIVERED TO YOU "AS IS," AND RENESAS MAKES NO REPRESENTATIONS OR WARRANTIES,
 * AND TO THE FULLEST EXTENT PERMISSIBLE UNDER APPLICABLE LAW, DISCLAIMS ALL WARRANTIES, WHETHER EXPLICITLY OR
 * IMPLICITLY, INCLUDING WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT, WITH
 * RESPECT TO THE SOFTWARE OR DOCUMENTATION. RENESAS SHALL HAVE NO LIABILITY ARISING OUT OF ANY SECURITY VULNERABILITY
 * OR BREACH. TO THE MAXIMUM EXTENT PERMITTED BY LAW, IN NO EVENT WILL RENESAS BE LIABLE TO YOU IN CONNECTION WITH THE
 * SOFTWARE OR DOCUMENTATION (OR ANY PERSON OR ENTITY CLAIMING RIGHTS DERIVED FROM YOU) FOR ANY LOSS, DAMAGE, OR
 * CLAIM, INCLUDING, BUT NOT LIMITED TO, LOST PROFITS, LOST DATA, OR INDIRECT, SPECIAL, CONSEQUENTIAL OR PUNITIVE
 * DAMAGES, REGARDLESS OF WHETHER SUCH CLAIM IS BASED ON CONTRACT, TORT, STRICT LIABILITY, OR OTHERWISE.
 **********************************************************************************************************************/

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>

/* Number of top predictions returned by postprocessing. */
#define TOP_K       (5)

/***********************************************************************************************************************
 * Function Name: dequantize_output_to_scores
 * Description  : Dequantizes an int8 output tensor to floating-point scores.
 *                dequantized_scores[i] = (quantized_output[i] - output_zero_point) * output_scale
 * Arguments    : p_quantized_output   - Pointer to int8 array of quantized model output values.
 *                _class_count         - Number of output classes.
 *                _output_scale        - Quantization scale factor.
 *                _output_zero_point   - Quantization zero-point offset.
 *                p_dequantized_scores - Pointer to float array to receive dequantized scores.
 * Return Value : None
 **********************************************************************************************************************/
void dequantize_output_to_scores(const int8_t *p_quantized_output,
                                 int _class_count,
                                 float _output_scale,
                                 int _output_zero_point,
                                 float *p_dequantized_scores);

/***********************************************************************************************************************
 * Function Name: postprocess
 * Description  : Dequantizes int8 logits and selects the top-k classes.
 *                Output arrays contain results sorted in descending score order.
 * Arguments    : p_quantized_output   - Pointer to int8 array of quantized model output values.
 *                _class_count         - Total number of output classes.
 *                _output_scale        - Quantization scale factor.
 *                _output_zero_point   - Quantization zero-point offset.
 *                _top_k_count         - Number of top results to return.
 *                p_top_indices        - Pointer to int array to receive top-k class indices.
 *                p_top_scores         - Pointer to float array to receive top-k scores.
 * Return Value : None
 **********************************************************************************************************************/
void postprocess(const int8_t *p_quantized_output,
                 int _class_count,
                 float _output_scale,
                 int _output_zero_point,
                 int _top_k_count,
                 int *p_top_indices,
                 float *p_top_scores);

#endif /* POSTPROCESSING_H */
