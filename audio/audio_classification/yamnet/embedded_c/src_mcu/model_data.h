#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>
#include <stddef.h>

/* TFLite model flatbuffer data */
extern const uint8_t g_model_data[];
extern const size_t g_model_data_size;

#define MODEL_DATA_SIZE 4194904

#endif /* MODEL_DATA_H */
