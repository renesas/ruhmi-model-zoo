/**
 * @file    postprocessing.h
 * @brief   Visual Wake Words (VWW, NPU) output postprocessing declarations - dequantization and top-1 person/not-person detection.
 *
 * @author  Renesas Electronics
 * @date    2025
 * @version 1.0
 */

#ifndef POSTPROCESSING_H
#define POSTPROCESSING_H

#include <stdint.h>
#include "../common/model_metadata.h"

/*
 * OUTPUT_SCALE, OUTPUT_ZP and OUTPUT_HAS_SOFTMAX are defined in model_metadata.h.
 * dequantize_output_to_scores() uses OUTPUT_SCALE and OUTPUT_ZP directly.
 */

/* Class index definitions */
#define CLASS_NOTPERSON      (0)
#define CLASS_PERSON         (1)
typedef struct
{
    uint32_t     top1_index;
    float        top1_score;
    const char  *p_top1_class_name;
} NN_Postprocess_Result_t;

/**
 * @brief Dequantize the raw int8 output logits to float p_scores.
 *
 * @param p_out_q              INT8 model output buffer [class_count]
 * @param class_count        Number of classes (MODEL_NUM_CLASSES)
 * @param p_scores             Output float buffer [class_count]
 */
void dequantize_output_to_scores(const int8_t *p_out_q,
                                 int           class_count,
                                 float        *p_scores);

/**
 * @brief Apply softmax in-place (required when OUTPUT_HAS_SOFTMAX == 0).
 * @param p_scores       float array of length class_count, modified in-place
 * @param class_count  number of elements
 */
void softmax_inplace(float *p_scores, int class_count);

/**
 * @brief Return the index of the highest value in p_scores[class_count].
 * @param p_scores       float array of length class_count
 * @param class_count  number of elements
 * @return             index of the highest score
 */
int argmax(const float *p_scores, int class_count);

/**
 * @brief Main postprocessing entry point for application flow.
 * @details Dequantizes output logits, applies softmax, and returns top-1 class.
 * @param[in] p_out_q           INT8 model output buffer [class_count].
 * @param[in] class_count     Number of output classes.
 * @param[out] p_scores         Float score buffer [class_count].
 * @param[out] p_predicted_class Top-1 class index.
 */
void postprocess(const int8_t *p_out_q,
                 int class_count,
                 float *p_scores,
                 int *p_predicted_class);

#endif /* POSTPROCESSING_H */
