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
