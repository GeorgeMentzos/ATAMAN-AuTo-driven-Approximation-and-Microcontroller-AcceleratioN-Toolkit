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
