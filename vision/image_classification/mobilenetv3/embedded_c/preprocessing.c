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

#include "../common/model_metadata.h"
#include "preprocessing.h"

/***********************************************************************************************************************
 * Function Name: preprocess
 * Description  : Preprocessing pipeline for MobileNetV3-Small INT8.
 *                Performs bilinear resize of the source image to model resolution (224x224),
 *                then quantizes each pixel to int8: q = clamp(pixel - 128, -128, 127).
 *                The model has a built-in Rescaling(1/255) layer so no external normalization is needed.
 * Arguments    : p_source_image_hwc      - Pointer to uint8 HWC source image [source_height x source_width x 3].
 *                _source_width           - Width of the source image in pixels.
 *                _source_height          - Height of the source image in pixels.
 *                p_quantized_input_tensor- Pointer to int8 HWC output tensor [224 x 224 x 3].
 * Return Value : None
 **********************************************************************************************************************/
void preprocess(const uint8_t *p_source_image_hwc,
                int _source_width,
                int _source_height,
                int8_t *p_quantized_input_tensor)
{
    const float _half_pixel = 0.5F;
    const float _one        = 1.0F;
    const int   _q_min      = -128;
    const int   _q_max      = 127;
    const float _scale_x = (float)_source_width / (float)MODEL_INPUT_W;
    const float _scale_y = (float)_source_height / (float)MODEL_INPUT_H;
    int _destination_row;

    /* Iterate over each destination row and map it back to source space. */
    for (_destination_row = 0; _destination_row < MODEL_INPUT_H; _destination_row++)
    {
        /* Half-pixel center convention (matches PIL bilinear behavior). */
        float _source_y_float = ((float)_destination_row + _half_pixel) * _scale_y - _half_pixel;
        int   _source_y0 = (int)_source_y_float;
        int   _source_y1 = _source_y0 + 1;
        float _weight_y = _source_y_float - (float)_source_y0;
        int _destination_col;

        /* Clamp vertical sampling coordinates to valid source bounds. */
        if (0 > _source_y0)
        {
            _source_y0 = 0;
        }
        if (_source_height <= _source_y1)
        {
            _source_y1 = _source_height - 1;
        }

        /* Iterate over each destination column and map it back to source space. */
        for (_destination_col = 0; _destination_col < MODEL_INPUT_W; _destination_col++)
        {
            float _source_x_float = ((float)_destination_col + _half_pixel) * _scale_x - _half_pixel;
            int   _source_x0 = (int)_source_x_float;
            int   _source_x1 = _source_x0 + 1;
            float _weight_x = _source_x_float - (float)_source_x0;
            int _channel_index;

            /* Clamp horizontal sampling coordinates to valid source bounds. */
            if (0 > _source_x0)
            {
                _source_x0 = 0;
            }
            if (_source_width <= _source_x1)
            {
                _source_x1 = _source_width - 1;
            }

            /* Process each color channel independently (HWC layout). */
            for (_channel_index = 0; _channel_index < MODEL_INPUT_C; _channel_index++)
            {
                /* Fetch the four neighboring source pixels used for interpolation. */
                float _p00 = (float)p_source_image_hwc[(_source_y0 * _source_width + _source_x0) * MODEL_INPUT_C + _channel_index];
                float _p10 = (float)p_source_image_hwc[(_source_y0 * _source_width + _source_x1) * MODEL_INPUT_C + _channel_index];
                float _p01 = (float)p_source_image_hwc[(_source_y1 * _source_width + _source_x0) * MODEL_INPUT_C + _channel_index];
                float _p11 = (float)p_source_image_hwc[(_source_y1 * _source_width + _source_x1) * MODEL_INPUT_C + _channel_index];

                /* Bilinear interpolation along X then Y. */
                float _top    = _p00 * (_one - _weight_x) + _p10 * _weight_x;
                float _bottom = _p01 * (_one - _weight_x) + _p11 * _weight_x;
                float _pixel  = _top  * (_one - _weight_y) + _bottom * _weight_y;

                /*
                 * MobileNetV3-Small quantization:
                 *   - Model has built-in Rescaling(1/255) layer
                 *   - INPUT_SCALE = 1.0, INPUT_ZP = -128
                 *   - q = clamp(round(pixel / INPUT_SCALE + INPUT_ZP), -128, 127)
                 *       = clamp(round(pixel - 128), -128, 127)
                 *
                 * No external normalization (divide by 127.5, subtract 1) is needed.
                 */
                float _quantized_float = _pixel / INPUT_SCALE + (float)INPUT_ZP;

                /* Round to nearest integer (ties away from zero), then clamp to int8. */
                int _quantized_int;
                if (0.0F <= _quantized_float)
                {
                    _quantized_int = (int)(_quantized_float + _half_pixel);
                }
                else
                {
                    _quantized_int = (int)(_quantized_float - _half_pixel);
                }

                if (_q_min > _quantized_int)
                {
                    _quantized_int = _q_min;
                }
                if (_q_max < _quantized_int)
                {
                    _quantized_int = _q_max;
                }

                /* Store quantized value into destination tensor (HWC layout). */
                p_quantized_input_tensor[(_destination_row * MODEL_INPUT_W + _destination_col) * MODEL_INPUT_C + _channel_index] = (int8_t)_quantized_int;
            }
        }
    }
}

