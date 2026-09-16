#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include "model.h"
#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"


/*
 * Tensor arena size (estimated). If AllocateTensors() fails, increase this.
 * Operator scratch buffers are hardware-dependent (CMSIS-NN vs reference)
 * and cannot be predicted exactly from Python.
 */
#define TENSOR_ARENA_SIZE 9249792
__attribute__((aligned(16))) static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

extern "C" {

/* TFLM globals */
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* model_input_tensor[1];
static TfLiteTensor* model_output_tensor[1];
static size_t g_arena_used_bytes = 0;
static int    g_last_invoke_status = 0;

int ModelInit(void) {
  static tflite::MicroMutableOpResolver<5> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddLogistic();
  resolver.AddReshape();
  resolver.AddMean();

  const tflite::Model* model = tflite::GetModel(g_model_data);
  if (model == nullptr) {
    return -1;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    return -2;
  }
  g_arena_used_bytes = interpreter->arena_used_bytes();

  /* Cache input/output tensor pointers */
  model_input_tensor[0] = interpreter->input(0);
  model_output_tensor[0] = interpreter->output(0);

  return 0;
}

void RunModel(bool clean_outputs) {
  if (clean_outputs) {
    memset(model_output_tensor[0]->data.raw, 0, model_output_tensor[0]->bytes);
  }

  g_last_invoke_status = (interpreter->Invoke() == kTfLiteOk) ? 0 : -1;
}

/* Arena debug getters */
size_t GetTensorArenaSizeBytes(void) { return (size_t)TENSOR_ARENA_SIZE; }
size_t GetTensorArenaUsedBytes(void) { return g_arena_used_bytes; }
int    GetLastInvokeStatus(void)     { return g_last_invoke_status; }

/* Input pointer accessors */
int16_t* GetModelInputPtr_serving_default_mel_patch_0() {
  return ((int16_t*)model_input_tensor[0]->data.raw);
}

/* Output pointer accessors */
int16_t* GetModelOutputPtr_StatefulPartitionedCall_0() {
  return ((int16_t*)model_output_tensor[0]->data.raw);
}

} /* extern "C" */
