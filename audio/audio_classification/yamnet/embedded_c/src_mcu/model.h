#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the TFLM interpreter. Call once before RunModel(). */
int ModelInit(void);

/* Run inference. Set clean_outputs=true to zero output buffer before run. */
void RunModel(bool clean_outputs);

/* Model input pointers - write your input data here before calling RunModel() */
int16_t* GetModelInputPtr_serving_default_mel_patch_0();

/* Model output pointers - read results here after RunModel() returns */
int16_t* GetModelOutputPtr_StatefulPartitionedCall_0();

/* Arena debug getters - available after ModelInit() returns 0 */
size_t GetTensorArenaSizeBytes(void);
size_t GetTensorArenaUsedBytes(void);
int    GetLastInvokeStatus(void);

/* Quantization parameters (scale / zero_point) for each quantized tensor */
#define INPUT_SCALE_SERVING_DEFAULT_MEL_PATCH_0          0.0002108143963f
#define INPUT_ZERO_POINT_SERVING_DEFAULT_MEL_PATCH_0      (0)
#define OUTPUT_SCALE_STATEFULPARTITIONEDCALL_0           3.051757812e-05f
#define OUTPUT_ZERO_POINT_STATEFULPARTITIONEDCALL_0       (0)

/* Quantization helpers
 * model_quantize  : float  -> int16_t  (use before writing input)
 * model_dequantize: int16_t -> float (use after reading output) */
static inline int16_t model_quantize(float x, float scale, int zero_point) {
    int v = (int)(x / scale) + zero_point;
    if (v < -32768) v = -32768;
    if (v >  32767) v = 32767;
    return (int16_t)v;
}
static inline float model_dequantize(int16_t x, float scale, int zero_point) {
    return ((float)(x) - (float)(zero_point)) * scale;
}

#ifdef __cplusplus
}
#endif

#endif /* MODEL_H */
