#ifndef PRECISIONTYPES_H
#define PRECISIONTYPES_H
#include "helper_math.h"
#include "double_math.h"

// precision for particle quanties (position, velocity, ...): choose float or double
typedef float FPpart;
typedef float2 FPpart2;
typedef float3 FPpart3;
typedef float4 FPpart4;

#define make_fppart2(...) make_float2(__VA_ARGS__)
#define make_fppart3(...) make_float3(__VA_ARGS__)
#define make_fppart4(...) make_float4(__VA_ARGS__)

// precision for field quantities (Ex, Bx, ...): choose float or double
typedef float FPfield;
typedef float2 FPfield2;
typedef float3 FPfield3;
typedef float4 FPfield4;

#define make_fpfield2(...) make_float2(__VA_ARGS__)
#define make_fpfield3(...) make_float3(__VA_ARGS__)
#define make_fpfield4(...) make_float4(__VA_ARGS__)

// precision for interpolated quanties (Ex, Bx, ...): choose float or double
typedef float FPinterp;
typedef float2 FPinterp2;
typedef float3 FPinterp3;
typedef float4 FPinterp4;

#define make_fpinterp2(...) make_float2(__VA_ARGS__)
#define make_fpinterp3(...) make_float3(__VA_ARGS__)
#define make_fpinterp4(...) make_float4(__VA_ARGS__)


#endif
