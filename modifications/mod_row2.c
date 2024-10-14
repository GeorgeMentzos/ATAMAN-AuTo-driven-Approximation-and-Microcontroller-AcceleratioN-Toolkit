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
