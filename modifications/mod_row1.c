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
