#ifndef MODEL_METADATA_H
#define MODEL_METADATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ══════════════════════════════════════════════════════════════════════
 * 1.  MODEL IDENTITY
 * ══════════════════════════════════════════════════════════════════════ */

#define MODEL_NAME          "MobileNetV3-Small"
#define MODEL_TASK          "Image Classification"
#define MODEL_DATASET       "ImageNet ILSVRC-2012"

/* ══════════════════════════════════════════════════════════════════════
 * 2.  INPUT TENSOR  —  [1, 192, 192, 3]  int8
 * ══════════════════════════════════════════════════════════════════════ */

#define MODEL_INPUT_H       192
#define MODEL_INPUT_W       192

#ifndef MODEL_CHANNELS
#define MODEL_INPUT_C       3
#else
#define MODEL_INPUT_C       MODEL_CHANNELS
#endif

#ifndef MODEL_INPUT_SIZE
#define MODEL_INPUT_SIZE    (MODEL_INPUT_H * MODEL_INPUT_W * MODEL_INPUT_C)
#endif

/** INT8 input quantization: q = clamp(round(pixel - 128), -128, 127) */
#define INPUT_SCALE         1.0f
#define INPUT_ZP            (-128)

/* ══════════════════════════════════════════════════════════════════════
 * 3.  OUTPUT TENSOR  —  [1, 1000]  int8
 * ══════════════════════════════════════════════════════════════════════ */

#define MODEL_OUTPUT_SIZE   1000

/** INT8 output dequantization: prob = (out_q - OUTPUT_ZP) * OUTPUT_SCALE */
#define OUTPUT_SCALE        0.00390625f   /* 1/256 */
#define OUTPUT_ZP           (-128)

/* ══════════════════════════════════════════════════════════════════════
 * 4.  NPU ARENA SIZE  (from sub_0000_tensors.h — kArenaSize_sub_0000)
 *     For CPU+NPU execution the arena is managed by the Ethos-U driver;
 *     kArenaSize_sub_0000 (258048 bytes) is the authoritative value.
 * ══════════════════════════════════════════════════════════════════════ */

#define CPU_RAM_TENSOR_ARENA_BYTES   258048   /* NPU arena: kArenaSize_sub_0000 */

#ifdef __cplusplus
}
#endif

#endif /* MODEL_METADATA_H */
