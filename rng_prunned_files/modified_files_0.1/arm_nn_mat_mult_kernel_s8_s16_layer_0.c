/*
 * SPDX-FileCopyrightText: Copyright 2010-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_nn_mat_mult_kernel_s8_s16.c
 * Description:  Matrix-multiplication function for convolution
 *
 * $Date:        5 Januray 2023
 * $Revision:    V.1.2.0
 *
 * Target :  Arm(R) M-Profile Architecture
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

// #include "perf.h"

/*
 * Matrix-multiplication function for convolution with per-channel requantization.
 *
 * Refer header file for details.
 *
 */

//KERNEL_NAME
int8_t *arm_nn_mat_mult_kernel_s8_s16_layer_0(
                                      const int8_t *input_a,
                                      const int16_t *input_b,
                                      const uint16_t output_ch,
                                      const int32_t *out_shift,
                                      const int32_t *out_mult,
                                      const int32_t out_offset,
                                      const int16_t activation_min,
                                      const int16_t activation_max,
                                      const uint16_t num_col_a,
                                      const int32_t *const output_bias,
                                      int8_t *out_0)
{

// global_increment();
// cycle_counter_begin();

#if !defined(ARM_MATH_MVEI)
    /* set up the second output pointers */
    int8_t *out_1 = out_0 + output_ch;
    const int32_t *bias = output_bias;

    uint16_t row_count = output_ch / 2;
    const int8_t *ip_a0 = input_a;
    /* this loop over rows in A */

//REMAINING_PRELOAD_START

//START_PRELOAD
/* setup pointers for B */
const int16_t *ip_b0 = input_b;
const int16_t *ip_b1 = ip_b0 + num_col_a;

int32_t ch_0_out_0 = 0;
int32_t ch_0_out_1 = 0;
int32_t ch_1_out_0 = 0;
int32_t ch_1_out_1 = 0;
/* Init accumulator with bias for channel N and N + 1 */
if (bias)
{
    ch_0_out_0 = *bias;
    ch_0_out_1 = *bias++;
    ch_1_out_0 = *bias;
    ch_1_out_1 = *bias++;
}
int32_t dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
int32_t dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-2490343, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-2490343, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(4194343, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(4194343, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1638424, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1638424, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(2687034, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(2687034, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-5963834, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-5963834, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(3932199, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(3932199, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-5636066, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-5636066, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-8257617, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-8257617, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(6881214, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(6881214, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(524218, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(524218, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(2949097, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(2949097, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(5832722, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(5832722, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-1834929, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-1834929, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(3473434, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(3473434, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-3538931, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-3538931, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1900552, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1900552, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-3211391, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-3211391, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-5308530, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-5308530, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(3407959, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(3407959, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(3342361, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(3342361, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(5701574, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(5701574, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1507357, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1507357, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(2424876, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(2424876, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-2031662, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-2031662, dsp_b1, ch_1_out_1);

int16_t b0 = *ip_b0++;
int16_t b1 = *ip_b1++;

//ROW1
ch_0_out_0 += 44 * b0;
ch_0_out_1 += 44 * b1;
//ROW2
ch_1_out_0 += -4 * b0;
ch_1_out_1 += -4 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += -2 * b0;
ch_0_out_1 += -2 * b1;
//ROW2
ch_1_out_0 += 18 * b0;
ch_1_out_1 += 18 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += -59 * b0;
ch_0_out_1 += -59 * b1;
//ROW2
ch_1_out_0 += -49 * b0;
ch_1_out_1 += -49 * b1;
ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
ch_0_out_0 += out_offset;
ch_0_out_0 = MAX(ch_0_out_0, activation_min);
ch_0_out_0 = MIN(ch_0_out_0, activation_max);
*out_0++ = (int8_t)ch_0_out_0;

ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
ch_0_out_1 += out_offset;
ch_0_out_1 = MAX(ch_0_out_1, activation_min);
ch_0_out_1 = MIN(ch_0_out_1, activation_max);
*out_1++ = (int8_t)ch_0_out_1;
out_mult++;
out_shift++;

ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
ch_1_out_0 += out_offset;
ch_1_out_0 = MAX(ch_1_out_0, activation_min);
ch_1_out_0 = MIN(ch_1_out_0, activation_max);
*out_0++ = (int8_t)ch_1_out_0;

ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
ch_1_out_1 += out_offset;
ch_1_out_1 = MAX(ch_1_out_1, activation_min);
ch_1_out_1 = MIN(ch_1_out_1, activation_max);
*out_1++ = (int8_t)ch_1_out_1;
out_mult++;
out_shift++;
/* setup pointers for B */
ip_b0 = input_b;
ip_b1 = ip_b0 + num_col_a;
ch_0_out_0 = 0;
ch_0_out_1 = 0;
ch_1_out_0 = 0;
ch_1_out_1 = 0;

/* Init accumulator with bias for channel N and N + 1 */
if (bias)
{
    ch_0_out_0 = *bias;
    ch_0_out_1 = *bias++;
    ch_1_out_0 = *bias;
    ch_1_out_1 = *bias++;
}
dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(3342373, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(3342373, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-5505066, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-5505066, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(6750312, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(6750312, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1703916, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1703916, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(8257663, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(8257663, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1114113, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1114113, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1245287, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1245287, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-3407952, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-3407952, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-5308475, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-5308475, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1900456, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1900456, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-3997756, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-3997756, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1965981, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1965981, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-5242929, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-5242929, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-5636211, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-5636211, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(589872, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(589872, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-5111866, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-5111866, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(2228253, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(2228253, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(3800981, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(3800981, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-4784127, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-4784127, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-3539039, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-3539039, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-2162764, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-2162764, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-8257607, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-8257607, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-4653124, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-4653124, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-589905, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-589905, dsp_b1, ch_1_out_1);

b0 = *ip_b0++;
b1 = *ip_b1++;

//ROW1
ch_0_out_0 += -109 * b0;
ch_0_out_1 += -109 * b1;
//ROW2
ch_1_out_0 += 73 * b0;
ch_1_out_1 += 73 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += -22 * b0;
ch_0_out_1 += -22 * b1;
//ROW2
ch_1_out_0 += 9 * b0;
ch_1_out_1 += 9 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += -30 * b0;
ch_0_out_1 += -30 * b1;
//ROW2
ch_1_out_0 += 33 * b0;
ch_1_out_1 += 33 * b1;
ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
ch_0_out_0 += out_offset;
ch_0_out_0 = MAX(ch_0_out_0, activation_min);
ch_0_out_0 = MIN(ch_0_out_0, activation_max);
*out_0++ = (int8_t)ch_0_out_0;

ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
ch_0_out_1 += out_offset;
ch_0_out_1 = MAX(ch_0_out_1, activation_min);
ch_0_out_1 = MIN(ch_0_out_1, activation_max);
*out_1++ = (int8_t)ch_0_out_1;
out_mult++;
out_shift++;

ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
ch_1_out_0 += out_offset;
ch_1_out_0 = MAX(ch_1_out_0, activation_min);
ch_1_out_0 = MIN(ch_1_out_0, activation_max);
*out_0++ = (int8_t)ch_1_out_0;

ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
ch_1_out_1 += out_offset;
ch_1_out_1 = MAX(ch_1_out_1, activation_min);
ch_1_out_1 = MIN(ch_1_out_1, activation_max);
*out_1++ = (int8_t)ch_1_out_1;
out_mult++;
out_shift++;
/* setup pointers for B */
ip_b0 = input_b;
ip_b1 = ip_b0 + num_col_a;
ch_0_out_0 = 0;
ch_0_out_1 = 0;
ch_1_out_0 = 0;
ch_1_out_1 = 0;

/* Init accumulator with bias for channel N and N + 1 */
if (bias)
{
    ch_0_out_0 = *bias;
    ch_0_out_1 = *bias++;
    ch_1_out_0 = *bias;
    ch_1_out_1 = *bias++;
}
dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(3538987, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(3538987, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1966030, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1966030, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-2817985, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-2817985, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1638433, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1638433, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-5242974, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-5242974, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(2359219, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(2359219, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(262169, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(262169, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-2490347, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-2490347, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(3211261, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(3211261, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1966092, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1966092, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(3407963, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(3407963, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-131111, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-131111, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-8257650, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-8257650, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-5570452, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-5570452, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(3145655, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(3145655, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(7208925, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(7208925, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1638469, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1638469, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-3801143, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-3801143, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(5898292, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(5898292, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-4980609, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-4980609, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-5767132, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-5767132, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(7208873, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(7208873, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-5963902, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-5963902, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-4259921, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-4259921, dsp_b1, ch_1_out_1);

b0 = *ip_b0++;
b1 = *ip_b1++;

//ROW1
ch_0_out_0 += 18 * b0;
ch_0_out_1 += 18 * b1;
//ROW2
ch_1_out_0 += 80 * b0;
ch_1_out_1 += 80 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += 56 * b0;
ch_0_out_1 += 56 * b1;
//ROW2
ch_1_out_0 += -36 * b0;
ch_1_out_1 += -36 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += 49 * b0;
ch_0_out_1 += 49 * b1;
//ROW2
ch_1_out_0 += -64 * b0;
ch_1_out_1 += -64 * b1;
ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
ch_0_out_0 += out_offset;
ch_0_out_0 = MAX(ch_0_out_0, activation_min);
ch_0_out_0 = MIN(ch_0_out_0, activation_max);
*out_0++ = (int8_t)ch_0_out_0;

ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
ch_0_out_1 += out_offset;
ch_0_out_1 = MAX(ch_0_out_1, activation_min);
ch_0_out_1 = MIN(ch_0_out_1, activation_max);
*out_1++ = (int8_t)ch_0_out_1;
out_mult++;
out_shift++;

ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
ch_1_out_0 += out_offset;
ch_1_out_0 = MAX(ch_1_out_0, activation_min);
ch_1_out_0 = MIN(ch_1_out_0, activation_max);
*out_0++ = (int8_t)ch_1_out_0;

ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
ch_1_out_1 += out_offset;
ch_1_out_1 = MAX(ch_1_out_1, activation_min);
ch_1_out_1 = MIN(ch_1_out_1, activation_max);
*out_1++ = (int8_t)ch_1_out_1;
out_mult++;
out_shift++;
/* setup pointers for B */
ip_b0 = input_b;
ip_b1 = ip_b0 + num_col_a;
ch_0_out_0 = 0;
ch_0_out_1 = 0;
ch_1_out_0 = 0;
ch_1_out_1 = 0;

/* Init accumulator with bias for channel N and N + 1 */
if (bias)
{
    ch_0_out_0 = *bias;
    ch_0_out_1 = *bias++;
    ch_1_out_0 = *bias;
    ch_1_out_1 = *bias++;
}
dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-982958, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-982958, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(327598, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(327598, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1834986, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1834986, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(2621532, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(2621532, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-2031685, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-2031685, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(7012344, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(7012344, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-2359299, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-2359299, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1572914, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1572914, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(917449, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(917449, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1114197, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1114197, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(786329, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(786329, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(5308365, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(5308365, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1048621, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1048621, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(327690, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(327690, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-1703878, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-1703878, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-8322982, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-8322982, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-262102, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-262102, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1245187, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1245187, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-8257643, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-8257643, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1834953, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1834953, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(3538858, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(3538858, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-5111736, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-5111736, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-1835034, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-1835034, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(393147, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(393147, dsp_b1, ch_1_out_1);

b0 = *ip_b0++;
b1 = *ip_b1++;

//ROW1
ch_0_out_0 += 69 * b0;
ch_0_out_1 += 69 * b1;
//ROW2
ch_1_out_0 += -63 * b0;
ch_1_out_1 += -63 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += 30 * b0;
ch_0_out_1 += 30 * b1;
//ROW2
ch_1_out_0 += -60 * b0;
ch_1_out_1 += -60 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += 61 * b0;
ch_0_out_1 += 61 * b1;
//ROW2
ch_1_out_0 += -59 * b0;
ch_1_out_1 += -59 * b1;
ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
ch_0_out_0 += out_offset;
ch_0_out_0 = MAX(ch_0_out_0, activation_min);
ch_0_out_0 = MIN(ch_0_out_0, activation_max);
*out_0++ = (int8_t)ch_0_out_0;

ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
ch_0_out_1 += out_offset;
ch_0_out_1 = MAX(ch_0_out_1, activation_min);
ch_0_out_1 = MIN(ch_0_out_1, activation_max);
*out_1++ = (int8_t)ch_0_out_1;
out_mult++;
out_shift++;

ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
ch_1_out_0 += out_offset;
ch_1_out_0 = MAX(ch_1_out_0, activation_min);
ch_1_out_0 = MIN(ch_1_out_0, activation_max);
*out_0++ = (int8_t)ch_1_out_0;

ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
ch_1_out_1 += out_offset;
ch_1_out_1 = MAX(ch_1_out_1, activation_min);
ch_1_out_1 = MIN(ch_1_out_1, activation_max);
*out_1++ = (int8_t)ch_1_out_1;
out_mult++;
out_shift++;
/* setup pointers for B */
ip_b0 = input_b;
ip_b1 = ip_b0 + num_col_a;
ch_0_out_0 = 0;
ch_0_out_1 = 0;
ch_1_out_0 = 0;
ch_1_out_1 = 0;

/* Init accumulator with bias for channel N and N + 1 */
if (bias)
{
    ch_0_out_0 = *bias;
    ch_0_out_1 = *bias++;
    ch_1_out_0 = *bias;
    ch_1_out_1 = *bias++;
}
dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-6160415, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-6160415, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(4063310, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(4063310, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-1245306, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-1245306, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-3014612, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-3014612, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-983036, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-983036, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-6160440, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-6160440, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(5242866, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(5242866, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-3538934, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-3538934, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-131042, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-131042, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-5046272, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-5046272, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-6291521, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-6291521, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-917568, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-917568, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-3801092, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-3801092, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-8257592, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-8257592, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(7733201, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(7733201, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(5898182, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(5898182, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(6160368, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(6160368, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(5701719, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(5701719, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-6881282, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-6881282, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-5767151, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-5767151, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1114047, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1114047, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(4652990, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(4652990, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-655398, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-655398, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(3735530, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(3735530, dsp_b1, ch_1_out_1);

b0 = *ip_b0++;
b1 = *ip_b1++;

//ROW1
ch_0_out_0 += 97 * b0;
ch_0_out_1 += 97 * b1;
//ROW2
ch_1_out_0 += 87 * b0;
ch_1_out_1 += 87 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += 120 * b0;
ch_0_out_1 += 120 * b1;
//ROW2
ch_1_out_0 += 9 * b0;
ch_1_out_1 += 9 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += 127 * b0;
ch_0_out_1 += 127 * b1;
//ROW2
ch_1_out_0 += 94 * b0;
ch_1_out_1 += 94 * b1;
ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
ch_0_out_0 += out_offset;
ch_0_out_0 = MAX(ch_0_out_0, activation_min);
ch_0_out_0 = MIN(ch_0_out_0, activation_max);
*out_0++ = (int8_t)ch_0_out_0;

ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
ch_0_out_1 += out_offset;
ch_0_out_1 = MAX(ch_0_out_1, activation_min);
ch_0_out_1 = MIN(ch_0_out_1, activation_max);
*out_1++ = (int8_t)ch_0_out_1;
out_mult++;
out_shift++;

ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
ch_1_out_0 += out_offset;
ch_1_out_0 = MAX(ch_1_out_0, activation_min);
ch_1_out_0 = MIN(ch_1_out_0, activation_max);
*out_0++ = (int8_t)ch_1_out_0;

ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
ch_1_out_1 += out_offset;
ch_1_out_1 = MAX(ch_1_out_1, activation_min);
ch_1_out_1 = MIN(ch_1_out_1, activation_max);
*out_1++ = (int8_t)ch_1_out_1;
out_mult++;
out_shift++;
/* setup pointers for B */
ip_b0 = input_b;
ip_b1 = ip_b0 + num_col_a;
ch_0_out_0 = 0;
ch_0_out_1 = 0;
ch_1_out_0 = 0;
ch_1_out_1 = 0;

/* Init accumulator with bias for channel N and N + 1 */
if (bias)
{
    ch_0_out_0 = *bias;
    ch_0_out_1 = *bias++;
    ch_1_out_0 = *bias;
    ch_1_out_1 = *bias++;
}
dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1048662, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1048662, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1048526, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1048526, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-1703875, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-1703875, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-6291424, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-6291424, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(3997755, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(3997755, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-8323035, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-8323035, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-2424877, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-2424877, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(2228273, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(2228273, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(7012378, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(7012378, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(786439, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(786439, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(4259954, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(4259954, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-7012261, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-7012261, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-4259946, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-4259946, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(6094805, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(6094805, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-1310788, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-1310788, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(4587408, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(4587408, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1507264, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1507264, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1441784, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1441784, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-2555895, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-2555895, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1900527, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1900527, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-8257620, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-8257620, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(3669899, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(3669899, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-4456548, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-4456548, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-5439399, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-5439399, dsp_b1, ch_1_out_1);

b0 = *ip_b0++;
b1 = *ip_b1++;

//ROW1
ch_0_out_0 += 20 * b0;
ch_0_out_1 += 20 * b1;
//ROW2
ch_1_out_0 += 68 * b0;
ch_1_out_1 += 68 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += 7 * b0;
ch_0_out_1 += 7 * b1;
//ROW2
ch_1_out_0 += 100 * b0;
ch_1_out_1 += 100 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += 67 * b0;
ch_0_out_1 += 67 * b1;
//ROW2
ch_1_out_0 += -80 * b0;
ch_1_out_1 += -80 * b1;
ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
ch_0_out_0 += out_offset;
ch_0_out_0 = MAX(ch_0_out_0, activation_min);
ch_0_out_0 = MIN(ch_0_out_0, activation_max);
*out_0++ = (int8_t)ch_0_out_0;

ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
ch_0_out_1 += out_offset;
ch_0_out_1 = MAX(ch_0_out_1, activation_min);
ch_0_out_1 = MIN(ch_0_out_1, activation_max);
*out_1++ = (int8_t)ch_0_out_1;
out_mult++;
out_shift++;

ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
ch_1_out_0 += out_offset;
ch_1_out_0 = MAX(ch_1_out_0, activation_min);
ch_1_out_0 = MIN(ch_1_out_0, activation_max);
*out_0++ = (int8_t)ch_1_out_0;

ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
ch_1_out_1 += out_offset;
ch_1_out_1 = MAX(ch_1_out_1, activation_min);
ch_1_out_1 = MIN(ch_1_out_1, activation_max);
*out_1++ = (int8_t)ch_1_out_1;
out_mult++;
out_shift++;
/* setup pointers for B */
ip_b0 = input_b;
ip_b1 = ip_b0 + num_col_a;
ch_0_out_0 = 0;
ch_0_out_1 = 0;
ch_1_out_0 = 0;
ch_1_out_1 = 0;

/* Init accumulator with bias for channel N and N + 1 */
if (bias)
{
    ch_0_out_0 = *bias;
    ch_0_out_1 = *bias++;
    ch_1_out_0 = *bias;
    ch_1_out_1 = *bias++;
}
dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-196655, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-196655, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1703925, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1703925, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-7733333, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-7733333, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1310779, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1310779, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(7077932, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(7077932, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1703944, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1703944, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(720894, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(720894, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1638351, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1638351, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(2031743, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(2031743, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-6422611, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-6422611, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(5898171, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(5898171, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(5505028, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(5505028, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(4456389, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(4456389, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(4718574, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(4718574, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-5439365, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-5439365, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1114059, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1114059, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(6291525, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(6291525, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-851895, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-851895, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-7208891, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-7208891, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-2883711, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-2883711, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1310756, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1310756, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-5636027, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-5636027, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-1638345, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-1638345, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(4521990, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(4521990, dsp_b1, ch_1_out_1);

b0 = *ip_b0++;
b1 = *ip_b1++;

//ROW1
ch_0_out_0 += 16 * b0;
ch_0_out_1 += 16 * b1;
//ROW2
ch_1_out_0 += -43 * b0;
ch_1_out_1 += -43 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += 81 * b0;
ch_0_out_1 += 81 * b1;
//ROW2
ch_1_out_0 += 31 * b0;
ch_1_out_1 += 31 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += -18 * b0;
ch_0_out_1 += -18 * b1;
//ROW2
ch_1_out_0 += 60 * b0;
ch_1_out_1 += 60 * b1;
ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
ch_0_out_0 += out_offset;
ch_0_out_0 = MAX(ch_0_out_0, activation_min);
ch_0_out_0 = MIN(ch_0_out_0, activation_max);
*out_0++ = (int8_t)ch_0_out_0;

ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
ch_0_out_1 += out_offset;
ch_0_out_1 = MAX(ch_0_out_1, activation_min);
ch_0_out_1 = MIN(ch_0_out_1, activation_max);
*out_1++ = (int8_t)ch_0_out_1;
out_mult++;
out_shift++;

ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
ch_1_out_0 += out_offset;
ch_1_out_0 = MAX(ch_1_out_0, activation_min);
ch_1_out_0 = MIN(ch_1_out_0, activation_max);
*out_0++ = (int8_t)ch_1_out_0;

ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
ch_1_out_1 += out_offset;
ch_1_out_1 = MAX(ch_1_out_1, activation_min);
ch_1_out_1 = MIN(ch_1_out_1, activation_max);
*out_1++ = (int8_t)ch_1_out_1;
out_mult++;
out_shift++;
/* setup pointers for B */
ip_b0 = input_b;
ip_b1 = ip_b0 + num_col_a;
ch_0_out_0 = 0;
ch_0_out_1 = 0;
ch_1_out_0 = 0;
ch_1_out_1 = 0;

/* Init accumulator with bias for channel N and N + 1 */
if (bias)
{
    ch_0_out_0 = *bias;
    ch_0_out_1 = *bias++;
    ch_1_out_0 = *bias;
    ch_1_out_1 = *bias++;
}
dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-5832777, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-5832777, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1310758, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1310758, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-8060976, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-8060976, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(4521990, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(4521990, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(2949084, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(2949084, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(5046213, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(5046213, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-65620, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-65620, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1572816, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1572816, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(7208961, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(7208961, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1834937, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1834937, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(5111874, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(5111874, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-4521987, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-4521987, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(65585, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(65585, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1572929, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1572929, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(7405634, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(7405634, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(4194271, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(4194271, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(8323122, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(8323122, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1703940, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1703940, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-5177270, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-5177270, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-8323014, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-8323014, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-1310810, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-1310810, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(917480, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(917480, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-1703930, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-1703930, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-5308444, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-5308444, dsp_b1, ch_1_out_1);

b0 = *ip_b0++;
b1 = *ip_b1++;

//ROW2
ch_1_out_0 += 84 * b0;
ch_1_out_1 += 84 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += -93 * b0;
ch_0_out_1 += -93 * b1;
//ROW2
ch_1_out_0 += -18 * b0;
ch_1_out_1 += -18 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += -98 * b0;
ch_0_out_1 += -98 * b1;
//ROW2
ch_1_out_0 += 60 * b0;
ch_1_out_1 += 60 * b1;
ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
ch_0_out_0 += out_offset;
ch_0_out_0 = MAX(ch_0_out_0, activation_min);
ch_0_out_0 = MIN(ch_0_out_0, activation_max);
*out_0++ = (int8_t)ch_0_out_0;

ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
ch_0_out_1 += out_offset;
ch_0_out_1 = MAX(ch_0_out_1, activation_min);
ch_0_out_1 = MIN(ch_0_out_1, activation_max);
*out_1++ = (int8_t)ch_0_out_1;
out_mult++;
out_shift++;

ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
ch_1_out_0 += out_offset;
ch_1_out_0 = MAX(ch_1_out_0, activation_min);
ch_1_out_0 = MIN(ch_1_out_0, activation_max);
*out_0++ = (int8_t)ch_1_out_0;

ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
ch_1_out_1 += out_offset;
ch_1_out_1 = MAX(ch_1_out_1, activation_min);
ch_1_out_1 = MIN(ch_1_out_1, activation_max);
*out_1++ = (int8_t)ch_1_out_1;
out_mult++;
out_shift++;
/* setup pointers for B */
ip_b0 = input_b;
ip_b1 = ip_b0 + num_col_a;
ch_0_out_0 = 0;
ch_0_out_1 = 0;
ch_1_out_0 = 0;
ch_1_out_1 = 0;

/* Init accumulator with bias for channel N and N + 1 */
if (bias)
{
    ch_0_out_0 = *bias;
    ch_0_out_1 = *bias++;
    ch_1_out_0 = *bias;
    ch_1_out_1 = *bias++;
}
dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1507256, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1507256, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-5505091, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-5505091, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-4849726, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-4849726, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-8257606, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-8257606, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1769544, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1769544, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-7209043, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-7209043, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-3604442, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-3604442, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-458751, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-458751, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(983089, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(983089, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(2228219, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(2228219, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-7929957, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-7929957, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-393221, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-393221, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-4784141, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-4784141, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1834981, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1834981, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-3211308, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-3211308, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-2097198, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-2097198, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-3670089, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-3670089, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-131155, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-131155, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-2686929, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-2686929, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(5963805, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(5963805, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-1769599, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-1769599, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(5242965, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(5242965, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-6619168, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-6619168, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(7864416, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(7864416, dsp_b1, ch_1_out_1);

b0 = *ip_b0++;
b1 = *ip_b1++;

//ROW1
ch_0_out_0 += 67 * b0;
ch_0_out_1 += 67 * b1;
//ROW2
ch_1_out_0 += 73 * b0;
ch_1_out_1 += 73 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += -73 * b0;
ch_0_out_1 += -73 * b1;
//ROW2
ch_1_out_0 += 67 * b0;
ch_1_out_1 += 67 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += -36 * b0;
ch_0_out_1 += -36 * b1;
//ROW2
ch_1_out_0 += -14 * b0;
ch_1_out_1 += -14 * b1;
ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
ch_0_out_0 += out_offset;
ch_0_out_0 = MAX(ch_0_out_0, activation_min);
ch_0_out_0 = MIN(ch_0_out_0, activation_max);
*out_0++ = (int8_t)ch_0_out_0;

ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
ch_0_out_1 += out_offset;
ch_0_out_1 = MAX(ch_0_out_1, activation_min);
ch_0_out_1 = MIN(ch_0_out_1, activation_max);
*out_1++ = (int8_t)ch_0_out_1;
out_mult++;
out_shift++;

ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
ch_1_out_0 += out_offset;
ch_1_out_0 = MAX(ch_1_out_0, activation_min);
ch_1_out_0 = MIN(ch_1_out_0, activation_max);
*out_0++ = (int8_t)ch_1_out_0;

ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
ch_1_out_1 += out_offset;
ch_1_out_1 = MAX(ch_1_out_1, activation_min);
ch_1_out_1 = MIN(ch_1_out_1, activation_max);
*out_1++ = (int8_t)ch_1_out_1;
out_mult++;
out_shift++;
/* setup pointers for B */
ip_b0 = input_b;
ip_b1 = ip_b0 + num_col_a;
ch_0_out_0 = 0;
ch_0_out_1 = 0;
ch_1_out_0 = 0;
ch_1_out_1 = 0;

/* Init accumulator with bias for channel N and N + 1 */
if (bias)
{
    ch_0_out_0 = *bias;
    ch_0_out_1 = *bias++;
    ch_1_out_0 = *bias;
    ch_1_out_1 = *bias++;
}
dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1441728, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1441728, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1048602, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1048602, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1179701, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1179701, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(4521905, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(4521905, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(3735598, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(3735598, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-7602156, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-7602156, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1769413, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1769413, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-851984, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-851984, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-5636219, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-5636219, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(3145751, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(3145751, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-2424801, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-2424801, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-4456438, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-4456438, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1441809, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1441809, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-851901, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-851901, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-1900502, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-1900502, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(589697, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(589697, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-5111881, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-5111881, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-2949106, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-2949106, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-786559, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-786559, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-3080144, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-3080144, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1703978, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1703978, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1179727, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1179727, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(4456509, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(4456509, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-4325349, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-4325349, dsp_b1, ch_1_out_1);

b0 = *ip_b0++;
b1 = *ip_b1++;

//ROW1
ch_0_out_0 += 16 * b0;
ch_0_out_1 += 16 * b1;
//ROW2
ch_1_out_0 += -1 * b0;
ch_1_out_1 += -1 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += 48 * b0;
ch_0_out_1 += 48 * b1;
//ROW2
ch_1_out_0 += -17 * b0;
ch_1_out_1 += -17 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += -23 * b0;
ch_0_out_1 += -23 * b1;
//ROW2
ch_1_out_0 += 7 * b0;
ch_1_out_1 += 7 * b1;
ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
ch_0_out_0 += out_offset;
ch_0_out_0 = MAX(ch_0_out_0, activation_min);
ch_0_out_0 = MIN(ch_0_out_0, activation_max);
*out_0++ = (int8_t)ch_0_out_0;

ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
ch_0_out_1 += out_offset;
ch_0_out_1 = MAX(ch_0_out_1, activation_min);
ch_0_out_1 = MIN(ch_0_out_1, activation_max);
*out_1++ = (int8_t)ch_0_out_1;
out_mult++;
out_shift++;

ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
ch_1_out_0 += out_offset;
ch_1_out_0 = MAX(ch_1_out_0, activation_min);
ch_1_out_0 = MIN(ch_1_out_0, activation_max);
*out_0++ = (int8_t)ch_1_out_0;

ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
ch_1_out_1 += out_offset;
ch_1_out_1 = MAX(ch_1_out_1, activation_min);
ch_1_out_1 = MIN(ch_1_out_1, activation_max);
*out_1++ = (int8_t)ch_1_out_1;
out_mult++;
out_shift++;
/* setup pointers for B */
ip_b0 = input_b;
ip_b1 = ip_b0 + num_col_a;
ch_0_out_0 = 0;
ch_0_out_1 = 0;
ch_1_out_0 = 0;
ch_1_out_1 = 0;

/* Init accumulator with bias for channel N and N + 1 */
if (bias)
{
    ch_0_out_0 = *bias;
    ch_0_out_1 = *bias++;
    ch_1_out_0 = *bias;
    ch_1_out_1 = *bias++;
}
dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(7864311, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(7864311, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-983069, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-983069, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-8126493, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-8126493, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(4259864, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(4259864, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(4456374, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(4456374, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(5832817, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(5832817, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1572809, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1572809, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(3342413, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(3342413, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(2293697, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(2293697, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-7995357, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-7995357, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-2686879, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-2686879, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-7208978, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-7208978, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-4915203, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-4915203, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-3342373, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-3342373, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-6029405, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-6029405, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(3997724, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(3997724, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-4194369, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-4194369, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(6291553, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(6291553, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-6357054, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-6357054, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-3014658, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-3014658, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(7536623, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(7536623, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-4325430, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-4325430, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1048698, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1048698, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-6357119, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-6357119, dsp_b1, ch_1_out_1);

b0 = *ip_b0++;
b1 = *ip_b1++;

//ROW1
ch_0_out_0 += 123 * b0;
ch_0_out_1 += 123 * b1;
//ROW2
ch_1_out_0 += -10 * b0;
ch_1_out_1 += -10 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += 127 * b0;
ch_0_out_1 += 127 * b1;
//ROW2
ch_1_out_0 += -35 * b0;
ch_1_out_1 += -35 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += 56 * b0;
ch_0_out_1 += 56 * b1;
//ROW2
ch_1_out_0 += -82 * b0;
ch_1_out_1 += -82 * b1;
ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
ch_0_out_0 += out_offset;
ch_0_out_0 = MAX(ch_0_out_0, activation_min);
ch_0_out_0 = MIN(ch_0_out_0, activation_max);
*out_0++ = (int8_t)ch_0_out_0;

ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
ch_0_out_1 += out_offset;
ch_0_out_1 = MAX(ch_0_out_1, activation_min);
ch_0_out_1 = MIN(ch_0_out_1, activation_max);
*out_1++ = (int8_t)ch_0_out_1;
out_mult++;
out_shift++;

ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
ch_1_out_0 += out_offset;
ch_1_out_0 = MAX(ch_1_out_0, activation_min);
ch_1_out_0 = MIN(ch_1_out_0, activation_max);
*out_0++ = (int8_t)ch_1_out_0;

ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
ch_1_out_1 += out_offset;
ch_1_out_1 = MAX(ch_1_out_1, activation_min);
ch_1_out_1 = MIN(ch_1_out_1, activation_max);
*out_1++ = (int8_t)ch_1_out_1;
out_mult++;
out_shift++;
/* setup pointers for B */
ip_b0 = input_b;
ip_b1 = ip_b0 + num_col_a;
ch_0_out_0 = 0;
ch_0_out_1 = 0;
ch_1_out_0 = 0;
ch_1_out_1 = 0;

/* Init accumulator with bias for channel N and N + 1 */
if (bias)
{
    ch_0_out_0 = *bias;
    ch_0_out_1 = *bias++;
    ch_1_out_0 = *bias;
    ch_1_out_1 = *bias++;
}
dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(4718605, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(4718605, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-130974, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-130974, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(65642, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(65642, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(7471148, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(7471148, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(7405551, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(7405551, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-4587468, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-4587468, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-1507446, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-1507446, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(917608, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(917608, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-720889, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-720889, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(4325339, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(4325339, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-524341, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-524341, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1769419, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1769419, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(2424820, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(2424820, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-5308427, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-5308427, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-2883510, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-2883510, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1638433, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1638433, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(5963799, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(5963799, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-6357089, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-6357089, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-720878, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-720878, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1245263, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1245263, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-2424870, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-2424870, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-7667825, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-7667825, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(5636030, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(5636030, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-8257602, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-8257602, dsp_b1, ch_1_out_1);

b0 = *ip_b0++;
b1 = *ip_b1++;

//ROW1
ch_0_out_0 += -127 * b0;
ch_0_out_1 += -127 * b1;
//ROW2
ch_1_out_0 += -64 * b0;
ch_1_out_1 += -64 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += -63 * b0;
ch_0_out_1 += -63 * b1;
//ROW2
ch_1_out_0 += -26 * b0;
ch_1_out_1 += -26 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += -32 * b0;
ch_0_out_1 += -32 * b1;
//ROW2
ch_1_out_0 += 2 * b0;
ch_1_out_1 += 2 * b1;
ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
ch_0_out_0 += out_offset;
ch_0_out_0 = MAX(ch_0_out_0, activation_min);
ch_0_out_0 = MIN(ch_0_out_0, activation_max);
*out_0++ = (int8_t)ch_0_out_0;

ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
ch_0_out_1 += out_offset;
ch_0_out_1 = MAX(ch_0_out_1, activation_min);
ch_0_out_1 = MIN(ch_0_out_1, activation_max);
*out_1++ = (int8_t)ch_0_out_1;
out_mult++;
out_shift++;

ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
ch_1_out_0 += out_offset;
ch_1_out_0 = MAX(ch_1_out_0, activation_min);
ch_1_out_0 = MIN(ch_1_out_0, activation_max);
*out_0++ = (int8_t)ch_1_out_0;

ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
ch_1_out_1 += out_offset;
ch_1_out_1 = MAX(ch_1_out_1, activation_min);
ch_1_out_1 = MIN(ch_1_out_1, activation_max);
*out_1++ = (int8_t)ch_1_out_1;
out_mult++;
out_shift++;
/* setup pointers for B */
ip_b0 = input_b;
ip_b1 = ip_b0 + num_col_a;
ch_0_out_0 = 0;
ch_0_out_1 = 0;
ch_1_out_0 = 0;
ch_1_out_1 = 0;

/* Init accumulator with bias for channel N and N + 1 */
if (bias)
{
    ch_0_out_0 = *bias;
    ch_0_out_1 = *bias++;
    ch_1_out_0 = *bias;
    ch_1_out_1 = *bias++;
}
dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(262067, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(262067, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1769553, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1769553, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-4325390, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-4325390, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(8323159, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(8323159, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(917608, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(917608, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(3145838, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(3145838, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(2228114, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(2228114, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(7208983, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(7208983, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-4980800, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-4980800, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1376258, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1376258, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(852030, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(852030, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(4259819, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(4259819, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(3342209, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(3342209, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-3211302, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-3211302, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-4784151, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-4784151, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(2686964, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(2686964, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-2424766, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-2424766, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-131051, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-131051, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(851853, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(851853, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-6946807, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-6946807, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-458766, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-458766, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-6881361, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-6881361, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(262223, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(262223, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-2031705, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-2031705, dsp_b1, ch_1_out_1);

b0 = *ip_b0++;
b1 = *ip_b1++;

//ROW1
ch_0_out_0 += -120 * b0;
ch_0_out_1 += -120 * b1;
//ROW2
ch_1_out_0 += -72 * b0;
ch_1_out_1 += -72 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += -34 * b0;
ch_0_out_1 += -34 * b1;
//ROW2
ch_1_out_0 += -102 * b0;
ch_1_out_1 += -102 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += 55 * b0;
ch_0_out_1 += 55 * b1;
//ROW2
ch_1_out_0 += -64 * b0;
ch_1_out_1 += -64 * b1;
ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
ch_0_out_0 += out_offset;
ch_0_out_0 = MAX(ch_0_out_0, activation_min);
ch_0_out_0 = MIN(ch_0_out_0, activation_max);
*out_0++ = (int8_t)ch_0_out_0;

ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
ch_0_out_1 += out_offset;
ch_0_out_1 = MAX(ch_0_out_1, activation_min);
ch_0_out_1 = MIN(ch_0_out_1, activation_max);
*out_1++ = (int8_t)ch_0_out_1;
out_mult++;
out_shift++;

ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
ch_1_out_0 += out_offset;
ch_1_out_0 = MAX(ch_1_out_0, activation_min);
ch_1_out_0 = MIN(ch_1_out_0, activation_max);
*out_0++ = (int8_t)ch_1_out_0;

ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
ch_1_out_1 += out_offset;
ch_1_out_1 = MAX(ch_1_out_1, activation_min);
ch_1_out_1 = MIN(ch_1_out_1, activation_max);
*out_1++ = (int8_t)ch_1_out_1;
out_mult++;
out_shift++;
/* setup pointers for B */
ip_b0 = input_b;
ip_b1 = ip_b0 + num_col_a;
ch_0_out_0 = 0;
ch_0_out_1 = 0;
ch_1_out_0 = 0;
ch_1_out_1 = 0;

/* Init accumulator with bias for channel N and N + 1 */
if (bias)
{
    ch_0_out_0 = *bias;
    ch_0_out_1 = *bias++;
    ch_1_out_0 = *bias;
    ch_1_out_1 = *bias++;
}
dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(5046191, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(5046191, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-6094785, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-6094785, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-3735611, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-3735611, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1376304, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1376304, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-3604433, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-3604433, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-5505094, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-5505094, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1638386, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1638386, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(786353, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(786353, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(3211208, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(3211208, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-7012298, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-7012298, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(2949247, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(2949247, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-3014618, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-3014618, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(5505045, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(5505045, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-6356942, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-6356942, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-1572903, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-1572903, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-8257587, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-8257587, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(3276865, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(3276865, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-6422587, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-6422587, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1507286, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1507286, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(589826, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(589826, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-3211344, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-3211344, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-5701668, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-5701668, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-3801135, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-3801135, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(2490375, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(2490375, dsp_b1, ch_1_out_1);

b0 = *ip_b0++;
b1 = *ip_b1++;

//ROW1
ch_0_out_0 += -22 * b0;
ch_0_out_1 += -22 * b1;
//ROW2
ch_1_out_0 += -67 * b0;
ch_1_out_1 += -67 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += 7 * b0;
ch_0_out_1 += 7 * b1;
//ROW2
ch_1_out_0 += -34 * b0;
ch_1_out_1 += -34 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += 23 * b0;
ch_0_out_1 += 23 * b1;
//ROW2
ch_1_out_0 += 6 * b0;
ch_1_out_1 += 6 * b1;
ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
ch_0_out_0 += out_offset;
ch_0_out_0 = MAX(ch_0_out_0, activation_min);
ch_0_out_0 = MIN(ch_0_out_0, activation_max);
*out_0++ = (int8_t)ch_0_out_0;

ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
ch_0_out_1 += out_offset;
ch_0_out_1 = MAX(ch_0_out_1, activation_min);
ch_0_out_1 = MIN(ch_0_out_1, activation_max);
*out_1++ = (int8_t)ch_0_out_1;
out_mult++;
out_shift++;

ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
ch_1_out_0 += out_offset;
ch_1_out_0 = MAX(ch_1_out_0, activation_min);
ch_1_out_0 = MIN(ch_1_out_0, activation_max);
*out_0++ = (int8_t)ch_1_out_0;

ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
ch_1_out_1 += out_offset;
ch_1_out_1 = MAX(ch_1_out_1, activation_min);
ch_1_out_1 = MIN(ch_1_out_1, activation_max);
*out_1++ = (int8_t)ch_1_out_1;
out_mult++;
out_shift++;
/* setup pointers for B */
ip_b0 = input_b;
ip_b1 = ip_b0 + num_col_a;
ch_0_out_0 = 0;
ch_0_out_1 = 0;
ch_1_out_0 = 0;
ch_1_out_1 = 0;

/* Init accumulator with bias for channel N and N + 1 */
if (bias)
{
    ch_0_out_0 = *bias;
    ch_0_out_1 = *bias++;
    ch_1_out_0 = *bias;
    ch_1_out_1 = *bias++;
}
dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-6815754, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-6815754, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-8126565, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-8126565, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1376206, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1376206, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(458675, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(458675, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(65598, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(65598, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-655416, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-655416, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(2228249, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(2228249, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(5898310, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(5898310, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-4063211, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-4063211, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-262041, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-262041, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-7012451, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-7012451, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(2359296, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(2359296, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(3145718, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(3145718, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(4587647, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(4587647, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(3080254, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(3080254, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(126, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(126, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(5898270, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(5898270, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-4718601, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-4718601, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-8257576, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-8257576, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(6160467, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(6160467, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-1507430, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-1507430, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-2228239, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-2228239, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-65503, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-65503, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1703959, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1703959, dsp_b1, ch_1_out_1);

b0 = *ip_b0++;
b1 = *ip_b1++;

//ROW1
ch_0_out_0 += 33 * b0;
ch_0_out_1 += 33 * b1;
//ROW2
ch_1_out_0 += -86 * b0;
ch_1_out_1 += -86 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += 94 * b0;
ch_0_out_1 += 94 * b1;
//ROW2
ch_1_out_0 += -105 * b0;
ch_1_out_1 += -105 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += 88 * b0;
ch_0_out_1 += 88 * b1;
//ROW2
ch_1_out_0 += -40 * b0;
ch_1_out_1 += -40 * b1;
ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
ch_0_out_0 += out_offset;
ch_0_out_0 = MAX(ch_0_out_0, activation_min);
ch_0_out_0 = MIN(ch_0_out_0, activation_max);
*out_0++ = (int8_t)ch_0_out_0;

ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
ch_0_out_1 += out_offset;
ch_0_out_1 = MAX(ch_0_out_1, activation_min);
ch_0_out_1 = MIN(ch_0_out_1, activation_max);
*out_1++ = (int8_t)ch_0_out_1;
out_mult++;
out_shift++;

ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
ch_1_out_0 += out_offset;
ch_1_out_0 = MAX(ch_1_out_0, activation_min);
ch_1_out_0 = MIN(ch_1_out_0, activation_max);
*out_0++ = (int8_t)ch_1_out_0;

ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
ch_1_out_1 += out_offset;
ch_1_out_1 = MAX(ch_1_out_1, activation_min);
ch_1_out_1 = MIN(ch_1_out_1, activation_max);
*out_1++ = (int8_t)ch_1_out_1;
out_mult++;
out_shift++;
/* setup pointers for B */
ip_b0 = input_b;
ip_b1 = ip_b0 + num_col_a;
ch_0_out_0 = 0;
ch_0_out_1 = 0;
ch_1_out_0 = 0;
ch_1_out_1 = 0;

/* Init accumulator with bias for channel N and N + 1 */
if (bias)
{
    ch_0_out_0 = *bias;
    ch_0_out_1 = *bias++;
    ch_1_out_0 = *bias;
    ch_1_out_1 = *bias++;
}
dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(982942, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(982942, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1310719, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1310719, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-3407999, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-3407999, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1572819, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1572819, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-5242834, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-5242834, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(786332, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(786332, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(8323147, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(8323147, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(524324, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(524324, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1376314, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1376314, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(3473450, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(3473450, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-5111760, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-5111760, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1572797, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1572797, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(851969, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(851969, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-8323053, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-8323053, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-6029316, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-6029316, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(524258, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(524258, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-589857, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-589857, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(1048551, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(1048551, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(1769556, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(1769556, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-1376210, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-1376210, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-4784034, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-4784034, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(2031656, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(2031656, dsp_b1, ch_1_out_1);

dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);
dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);

//ROW1
ch_0_out_0 = SMLAD(-1703883, dsp_b0, ch_0_out_0);
ch_0_out_1 = SMLAD(-1703883, dsp_b1, ch_0_out_1);
//ROW2
ch_1_out_0 = SMLAD(-2293848, dsp_b0, ch_1_out_0);
ch_1_out_1 = SMLAD(-2293848, dsp_b1, ch_1_out_1);

b0 = *ip_b0++;
b1 = *ip_b1++;

//ROW1
ch_0_out_0 += -29 * b0;
ch_0_out_1 += -29 * b1;
//ROW2
ch_1_out_0 += 14 * b0;
ch_1_out_1 += 14 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += -54 * b0;
ch_0_out_1 += -54 * b1;
//ROW2
ch_1_out_0 += -2 * b0;
ch_1_out_1 += -2 * b1;
//POINTER_ITERATION
b0 = *ip_b0++;
b1 = *ip_b1++;
//ROW1
ch_0_out_0 += -104 * b0;
ch_0_out_1 += -104 * b1;
//ROW2
ch_1_out_0 += 60 * b0;
ch_1_out_1 += 60 * b1;
ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
ch_0_out_0 += out_offset;
ch_0_out_0 = MAX(ch_0_out_0, activation_min);
ch_0_out_0 = MIN(ch_0_out_0, activation_max);
*out_0++ = (int8_t)ch_0_out_0;

ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
ch_0_out_1 += out_offset;
ch_0_out_1 = MAX(ch_0_out_1, activation_min);
ch_0_out_1 = MIN(ch_0_out_1, activation_max);
*out_1++ = (int8_t)ch_0_out_1;
out_mult++;
out_shift++;

ch_1_out_0 = arm_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
ch_1_out_0 += out_offset;
ch_1_out_0 = MAX(ch_1_out_0, activation_min);
ch_1_out_0 = MIN(ch_1_out_0, activation_max);
*out_0++ = (int8_t)ch_1_out_0;

ch_1_out_1 = arm_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
ch_1_out_1 += out_offset;
ch_1_out_1 = MAX(ch_1_out_1, activation_min);
ch_1_out_1 = MIN(ch_1_out_1, activation_max);
*out_1++ = (int8_t)ch_1_out_1;
out_mult++;
out_shift++;
//END_PRELOAD

    /* compute the last odd numbered row if any */
    if (output_ch & 0x1)
    {
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + num_col_a;

        int32_t ch_0_out_0 = 0;
        int32_t ch_0_out_1 = 0;

        /* load the bias */
        if (bias)
        {
            ch_0_out_0 = *bias;
            ch_0_out_1 = *bias++;
        }

    #if defined(ARM_MATH_DSP)
        uint16_t col_count = num_col_a >> 2;
        while (col_count)
        {
            int32_t a01, a02;
            int32_t b0 = arm_nn_read_q15x2_ia(&ip_b0);
            int32_t b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ip_a0 = read_and_pad(ip_a0, &a01, &a02);

            ch_0_out_0 = SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = SMLAD(a01, b1, ch_0_out_1);

            b0 = arm_nn_read_q15x2_ia(&ip_b0);
            b1 = arm_nn_read_q15x2_ia(&ip_b1);
            ch_0_out_0 = SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = SMLAD(a02, b1, ch_0_out_1);

            col_count--;
        }
        col_count = num_col_a & 0x3;
    #else
        uint16_t col_count = num_col_a;
    #endif

//START_PRELOAD_ODD
        while (col_count)
        {
            int8_t a0 = *ip_a0++;
            int16_t b0 = *ip_b0++;
            int16_t b1 = *ip_b1++;

            ch_0_out_0 += a0 * b0;
            ch_0_out_1 += a0 * b1;
            col_count--;
        }
//END_PRELOAD_ODD

        ch_0_out_0 = arm_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        *out_0++ = (int8_t)ch_0_out_0;

        ch_0_out_1 = arm_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
        ch_0_out_1 += out_offset;
        ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        *out_1++ = (int8_t)ch_0_out_1;
        out_mult++;
        out_shift++;
    }

    out_0 += output_ch;

    // cycle_counter_end();
    /* return the new output pointer with offset */
    return out_0;
#else
    (void)input_a;
    (void)input_b;
    (void)output_ch;
    (void)out_shift;
    (void)out_mult;
    (void)out_offset;
    (void)activation_min;
    (void)activation_max;
    (void)num_col_a;
    (void)output_bias;
    (void)out_0;
    /* To be completed */
    return NULL;
#endif
}
