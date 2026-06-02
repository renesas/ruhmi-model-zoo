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

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdint.h>

/***********************************************************************************************************************
 * Function Name: preprocess
 * Description  : Preprocessing pipeline for MobileNetV3-Small INT8.
 *                Performs bilinear resize of source image to model resolution (224x224),
 *                then quantizes to int8: q = clamp(pixel - 128, -128, 127).
 *                The model has a built-in Rescaling(1/255) layer, so no external normalization is needed.
 * Arguments    : p_source_image_hwc      - Pointer to uint8 HWC source image [source_height x source_width x 3].
 *                _source_width           - Width of the source image in pixels.
 *                _source_height          - Height of the source image in pixels.
 *                p_quantized_input_tensor- Pointer to int8 HWC output tensor [224 x 224 x 3].
 * Return Value : None
 **********************************************************************************************************************/
void preprocess(const uint8_t *p_source_image_hwc,
                int _source_width,
                int _source_height,
                int8_t *p_quantized_input_tensor);

#endif /* PREPROCESSING_H */
