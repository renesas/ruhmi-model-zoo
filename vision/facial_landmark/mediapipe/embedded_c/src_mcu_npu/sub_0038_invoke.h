#ifndef __SUB_0038_INVOKE_H__
#define __SUB_0038_INVOKE_H__

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

// Declare arenas
extern uint8_t sub_0038_arena[864];

// Fast scratch arena not used for Ethos-U55
// We will not create it for now and reuse the address of the other arena
extern uint8_t* sub_0038_fast_scratch; // size: 864

int sub_0038_invoke(bool clean_outputs);


#endif // __SUB_0038_INVOKE_H__
