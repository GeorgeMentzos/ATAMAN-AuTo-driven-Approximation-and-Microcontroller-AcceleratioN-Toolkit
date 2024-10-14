    {
        /* setup pointers for B */
        const int16_t *ip_b0 = input_b;
        const int16_t *ip_b1 = ip_b0 + num_col_a;

        /* align the second pointer for A */
        const int8_t *ip_a1 = ip_a0 + num_col_a;

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

    #if defined(ARM_MATH_DSP)
        uint16_t col_count = num_col_a / 4;
        /* accumulate over the vector */
        while (col_count)
        {
            int32_t a01, a02, a11, a12;
            int32_t b0 = arm_nn_read_q15x2_ia(&ip_b0);
            int32_t b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ip_a0 = read_and_pad(ip_a0, &a01, &a02);
            ip_a1 = read_and_pad(ip_a1, &a11, &a12);

            ch_0_out_0 = SMLAD(a01, b0, ch_0_out_0);
            ch_0_out_1 = SMLAD(a01, b1, ch_0_out_1);
            ch_1_out_0 = SMLAD(a11, b0, ch_1_out_0);
            ch_1_out_1 = SMLAD(a11, b1, ch_1_out_1);

            b0 = arm_nn_read_q15x2_ia(&ip_b0);
            b1 = arm_nn_read_q15x2_ia(&ip_b1);

            ch_0_out_0 = SMLAD(a02, b0, ch_0_out_0);
            ch_0_out_1 = SMLAD(a02, b1, ch_0_out_1);
            ch_1_out_0 = SMLAD(a12, b0, ch_1_out_0);
            ch_1_out_1 = SMLAD(a12, b1, ch_1_out_1);

            col_count--;
        } /* while over col_count */
        col_count = num_col_a & 0x3;
    #else
        uint16_t col_count = num_col_a;
    #endif

        while (col_count)
        {
            int8_t a0 = *ip_a0++;
            int16_t b0 = *ip_b0++;
            int8_t a1 = *ip_a1++;
            int16_t b1 = *ip_b1++;

            ch_0_out_0 += a0 * b0;
            ch_0_out_1 += a0 * b1;
            ch_1_out_0 += a1 * b0;
            ch_1_out_1 += a1 * b1;
            col_count--;
        } /* while over col_count */

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

        /* skip row */
        ip_a0 += num_col_a;
        row_count--;
    }
//END_PRELOAD
