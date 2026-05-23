// Standalone CUDA executable for profiling kerr_geodesic_kernel with Nsight Compute.
// Compile: nvcc -o profile_geodesic profile_geodesic.cu -arch=sm_89
// Profile: ncu ./profile_geodesic

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Minimal kernel wrapper matching cuda/kernels.cu interface
__device__ float clamp_spin_d(float a) {
    if (a < -0.999999f) a = -0.999999f;
    if (a >  0.999999f) a =  0.999999f;
    return a;
}

__device__ float sigma_d(float r, float theta, float a) {
    a = clamp_spin_d(a);
    return r * r + a * a * cosf(theta) * cosf(theta);
}

__device__ float delta_d(float r, float a) {
    a = clamp_spin_d(a);
    return r * r - 2.0f * r + a * a;
}

__device__ float horizon_radius_d(float a) {
    a = clamp_spin_d(a);
    return 1.0f + sqrtf(fmaxf(0.0f, 1.0f - a * a));
}

__device__ float isco_radius_d(float a) {
    a = clamp_spin_d(a);
    float one = 1.0f;
    float am1 = 1.0f - a;
    float ap1 = 1.0f + a;
    float cbrtam1 = cbrtf(am1);
    float cbrt_ap1 = cbrtf(ap1);
    float cbrt_oma2 = cbrtf(one - a * a);
    float z1 = 1.0f + cbrt_oma2 * (cbrt_ap1 + cbrtam1);
    float z2 = sqrtf(3.0f * a * a + z1 * z1);
    float sign = (a >= 0.0f) ? 1.0f : -1.0f;
    return 3.0f + z2 - sign * sqrtf((3.0f - z1) * (3.0f + z1 + 2.0f * z2));
}

__device__ void inverse_metric_terms_d(float r, float theta, float a,
    float* gtt, float* gtphi, float* grr, float* gtheta, float* gphi) {
    a = clamp_spin_d(a);
    float th = theta;
    if (th < 1.0e-7f) th = 1.0e-7f;
    if (th > 3.141592653589793f - 1.0e-7f) th = 3.141592653589793f - 1.0e-7f;
    float rr = fmaxf(r, horizon_radius_d(a) + 1.0e-7f);
    float aa = a * a;
    float s = sinf(th);
    float c = cosf(th);
    float sin2 = fmaxf(s * s, 1.0e-12f);
    float sig = rr * rr + aa * c * c;
    float dele = fmaxf(rr * rr - 2.0f * rr + aa, 1.0e-12f);
    float r2_a2 = rr * rr + aa;
    float big_a = r2_a2 * r2_a2 - aa * dele * sin2;
    float denom = sig * dele;
    float gph_n = dele - aa * sin2;
    float gph_d = denom * sin2;
    *gtt = -big_a / denom;
    *gtphi = (-2.0f * a * rr) / denom;
    *grr = dele / sig;
    *gtheta = 1.0f / sig;
    *gphi = gph_n / gph_d;
}

__device__ void metric_derivatives_d(float r, float theta, float a,
    float* dgtt_r, float* dgtphi_r, float* dgrr_r, float* dgtheta_r, float* dgphi_r,
    float* dgtt_t, float* dgtphi_t, float* dgrr_t, float* dgtheta_t, float* dgphi_t) {
    a = clamp_spin_d(a);
    float th = theta;
    if (th < 1.0e-7f) th = 1.0e-7f;
    if (th > 3.141592653589793f - 1.0e-7f) th = 3.141592653589793f - 1.0e-7f;
    float rr = fmaxf(r, horizon_radius_d(a) + 1.0e-7f);
    float aa = a * a;
    float s = sinf(th);
    float c = cosf(th);
    float sin2 = fmaxf(s * s, 1.0e-12f);
    float dsin2_dt = 2.0f * s * c;
    float sig = rr * rr + aa * c * c;
    float sig_r = 2.0f * rr;
    float sig_t = -2.0f * aa * s * c;
    float dele = fmaxf(rr * rr - 2.0f * rr + aa, 1.0e-12f);
    float dele_r = 2.0f * rr - 2.0f;
    float r2_a2 = rr * rr + aa;
    float big_a = r2_a2 * r2_a2 - aa * dele * sin2;
    float big_a_r = 4.0f * rr * r2_a2 - aa * dele_r * sin2;
    float big_a_t = -aa * dele * dsin2_dt;
    float denom = sig * dele;
    float denom_r = sig_r * dele + sig * dele_r;
    float denom_t = sig_t * dele;
    float denom2 = denom * denom;
    float sig2 = sig * sig;
    *dgtt_r = -((big_a_r * denom - big_a * denom_r) / denom2);
    *dgtt_t = -((big_a_t * denom - big_a * denom_t) / denom2);
    float gtphi_n = -2.0f * a * rr;
    float gtphi_n_r = -2.0f * a;
    *dgtphi_r = (gtphi_n_r * denom - gtphi_n * denom_r) / denom2;
    *dgtphi_t = -(gtphi_n * denom_t) / denom2;
    *dgrr_r = (dele_r * sig - dele * sig_r) / sig2;
    *dgrr_t = -(dele * sig_t) / sig2;
    *dgtheta_r = -sig_r / sig2;
    *dgtheta_t = -sig_t / sig2;
    float gph_n = dele - aa * sin2;
    float gph_n_r = dele_r;
    float gph_n_t = -aa * dsin2_dt;
    float gph_d = denom * sin2;
    float gph_d_r = denom_r * sin2;
    float gph_d_t = denom_t * sin2 + denom * dsin2_dt;
    float gph_d2 = gph_d * gph_d;
    *dgphi_r = (gph_n_r * gph_d - gph_n * gph_d_r) / gph_d2;
    *dgphi_t = (gph_n_t * gph_d - gph_n * gph_d_t) / gph_d2;
}

__device__ void hamiltonian_rhs_d(const float* state, float a, float* out) {
    float r = state[1];
    float theta = state[2];
    if (theta < 1.0e-6f) theta = 1.0e-6f;
    if (theta > 3.141592653589793f - 1.0e-6f) theta = 3.141592653589793f - 1.0e-6f;
    float pt = state[4];
    float pr = state[5];
    float ptheta = state[6];
    float pphi = state[7];
    float gtt, gtphi, grr, gtheta, gphi;
    inverse_metric_terms_d(r, theta, a, &gtt, &gtphi, &grr, &gtheta, &gphi);
    float dx_t = gtt * pt + gtphi * pphi;
    float dx_r = grr * pr;
    float dx_theta = gtheta * ptheta;
    float dx_phi = gtphi * pt + gphi * pphi;
    float dgtt_r, dgtphi_r, dgrr_r, dgtheta_r, dgphi_r;
    float dgtt_t, dgtphi_t, dgrr_t, dgtheta_t, dgphi_t;
    metric_derivatives_d(r, theta, a,
        &dgtt_r, &dgtphi_r, &dgrr_r, &dgtheta_r, &dgphi_r,
        &dgtt_t, &dgtphi_t, &dgrr_t, &dgtheta_t, &dgphi_t);
    float dp_r = -0.5f * (
        dgtt_r * pt * pt + 2.0f * dgtphi_r * pt * pphi
      + dgrr_r * pr * pr + dgtheta_r * ptheta * ptheta + dgphi_r * pphi * pphi
    );
    float dp_theta = -0.5f * (
        dgtt_t * pt * pt + 2.0f * dgtphi_t * pt * pphi
      + dgrr_t * pr * pr + dgtheta_t * ptheta * ptheta + dgphi_t * pphi * pphi
    );
    if (!isfinite(dp_r)) dp_r = 1.0e20f;
    if (!isfinite(dp_theta)) dp_theta = 1.0e20f;
    out[0] = dx_t;
    out[1] = dx_r;
    out[2] = dx_theta;
    out[3] = dx_phi;
    out[4] = 0.0f;
    out[5] = dp_r;
    out[6] = dp_theta;
    out[7] = 0.0f;
}

__device__ void rk4_step_d(float* state, float a, float h) {
    float k1[8], k2[8], k3[8], k4[8], tmp[8];
    hamiltonian_rhs_d(state, a, k1);
    for (int i = 0; i < 8; ++i) tmp[i] = state[i] + 0.5f * h * k1[i];
    hamiltonian_rhs_d(tmp, a, k2);
    for (int i = 0; i < 8; ++i) tmp[i] = state[i] + 0.5f * h * k2[i];
    hamiltonian_rhs_d(tmp, a, k3);
    for (int i = 0; i < 8; ++i) tmp[i] = state[i] + h * k3[i];
    hamiltonian_rhs_d(tmp, a, k4);
    for (int i = 0; i < 8; ++i) {
        state[i] = state[i] + (h / 6.0f) * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
    }
}

__device__ void metric_covariant_d(float r, float theta, float a,
    float* g00, float* g03, float* g11, float* g22, float* g33) {
    a = clamp_spin_d(a);
    float th = theta;
    if (th < 1.0e-7f) th = 1.0e-7f;
    if (th > 3.141592653589793f - 1.0e-7f) th = 3.141592653589793f - 1.0e-7f;
    float rr = fmaxf(r, horizon_radius_d(a) + 1.0e-7f);
    float aa = a * a;
    float s = sinf(th);
    float sig = rr * rr + aa * cosf(th) * cosf(th);
    float sin2 = s * s;
    *g00 = -(1.0f - 2.0f * rr / sig);
    *g03 = -2.0f * a * rr * sin2 / sig;
    *g11 = sig / fmaxf(delta_d(rr, a), 1.0e-12f);
    *g22 = sig;
    *g33 = (rr * rr + aa + 2.0f * aa * rr * sin2 / sig) * sin2;
}

__device__ void metric_contravariant_d(float r, float theta, float a,
    float* gi00, float* gi03, float* gi11, float* gi22, float* gi33) {
    a = clamp_spin_d(a);
    float th = theta;
    if (th < 1.0e-7f) th = 1.0e-7f;
    if (th > 3.141592653589793f - 1.0e-7f) th = 3.141592653589793f - 1.0e-7f;
    float rr = fmaxf(r, horizon_radius_d(a) + 1.0e-7f);
    float aa = a * a;
    float s = sinf(th);
    float sin2 = fmaxf(s * s, 1.0e-12f);
    float dele = fmaxf(delta_d(rr, a), 1.0e-12f);
    float r2_a2 = rr * rr + aa;
    float big_a = r2_a2 * r2_a2 - aa * dele * sin2;
    float denom = sigma_d(rr, th, a) * dele;
    *gi00 = -big_a / denom;
    *gi03 = -2.0f * a * rr / denom;
    *gi11 = dele / sigma_d(rr, th, a);
    *gi22 = 1.0f / sigma_d(rr, th, a);
    *gi33 = (dele - aa * sin2) / (denom * sin2);
}

__device__ void initial_photon_state_d(float alpha, float beta, float a, float r_obs, float inclination, float* state) {
    a = clamp_spin_d(a);
    float theta = inclination;
    if (theta < 1.0e-5f) theta = 1.0e-5f;
    if (theta > 3.141592653589793f - 1.0e-5f) theta = 3.141592653589793f - 1.0e-5f;
    float n_phi = alpha / r_obs;
    float n_theta = beta / r_obs;
    float transverse2 = n_phi * n_phi + n_theta * n_theta;
    if (transverse2 >= 0.95f) {
        float scale = sqrtf(0.95f / transverse2);
        n_phi *= scale;
        n_theta *= scale;
        transverse2 = n_phi * n_phi + n_theta * n_theta;
    }
    float n_r = -sqrtf(fmaxf(1.0e-12f, 1.0f - transverse2));
    float g00, g03, g11, g22, g33;
    metric_covariant_d(r_obs, theta, a, &g00, &g03, &g11, &g22, &g33);
    float p_r = sqrtf(g11) * n_r;
    float p_theta = sqrtf(g22) * n_theta;
    float p_phi = sqrtf(fmaxf(g33, 1.0e-12f)) * n_phi;
    float gi00, gi03, gi11, gi22, gi33;
    metric_contravariant_d(r_obs, theta, a, &gi00, &gi03, &gi11, &gi22, &gi33);
    float A = gi00;
    float B = 2.0f * gi03 * p_phi;
    float C = gi11 * p_r * p_r + gi22 * p_theta * p_theta + gi33 * p_phi * p_phi;
    float disc = fmaxf(0.0f, B * B - 4.0f * A * C);
    float sqrt_disc = sqrtf(disc);
    float root1 = (-B + sqrt_disc) / (2.0f * A);
    float root2 = (-B - sqrt_disc) / (2.0f * A);
    float future_dt1 = gi00 * root1 + gi03 * p_phi;
    float future_dt2 = gi00 * root2 + gi03 * p_phi;
    float p_t;
    if (future_dt1 > 0.0f && future_dt2 > 0.0f) {
        p_t = (root1 < root2) ? root1 : root2;
    } else if (future_dt1 > 0.0f) {
        p_t = root1;
    } else if (future_dt2 > 0.0f) {
        p_t = root2;
    } else {
        p_t = (root1 < root2) ? root1 : root2;
    }
    state[0] = 0.0f;
    state[1] = r_obs;
    state[2] = theta;
    state[3] = 0.0f;
    state[4] = p_t;
    state[5] = p_r;
    state[6] = p_theta;
    state[7] = p_phi;
}

__device__ float keplerian_omega_d(float r, float a) {
    a = clamp_spin_d(a);
    return 1.0f / (powf(r, 1.5f) + a);
}

__device__ float redshift_factor_d(float r, float theta, float a, float lambda_photon) {
    a = clamp_spin_d(a);
    float omega = keplerian_omega_d(r, a);
    float g00, g03, g11, g22, g33;
    metric_covariant_d(r, theta, a, &g00, &g03, &g11, &g22, &g33);
    float norm = -(g00 + 2.0f * omega * g03 + omega * omega * g33);
    float u_t = 1.0f / sqrtf(fmaxf(norm, 1.0e-12f));
    float denom = u_t * fmaxf(0.05f, 1.0f - omega * lambda_photon);
    float gfac = 1.0f / denom;
    if (gfac < 0.02f) gfac = 0.02f;
    if (gfac > 8.0f) gfac = 8.0f;
    return gfac;
}

__device__ float hit_intensity_d(float r, float theta, float a, float lambda_photon, float q, int emission_model, float r_in) {
    float emitted;
    if (emission_model == 1) {
        float rr = fmaxf(r, r_in);
        emitted = powf(rr, -3.0f) * fmaxf(0.0f, 1.0f - sqrtf(r_in / rr));
    } else {
        emitted = powf(fmaxf(r, 1.0e-6f), -q);
    }
    float g = redshift_factor_d(r, theta, a, lambda_photon);
    return emitted * g * g * g;
}

__global__ void kerr_geodesic_kernel(
    const int width, const int height,
    const float spin, const float inclination_rad, const float fov_m,
    const float r_inner, const float r_outer, const float emissivity_q,
    const int emission_model, const float r_obs,
    const float step_size, const int max_steps,
    const float horizon_epsilon, const float escape_radius,
    float* intensity, float* redshift, float* temperature,
    unsigned char* hit_mask, unsigned char* status_code, float* null_error
) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= width || iy >= height) return;
    const int idx = iy * width + ix;
    const float denom_x = (float)(width > 1 ? width - 1 : 1);
    const float denom_y = (float)(height > 1 ? height - 1 : 1);
    const float alpha = ((float)ix / denom_x - 0.5f) * fov_m;
    const float beta  = ((float)iy / denom_y - 0.5f) * fov_m;
    float state[8];
    initial_photon_state_d(alpha, beta, spin, r_obs, inclination_rad, state);
    float a = spin;
    float r_h = horizon_radius_d(a);
    float r_in = r_inner;
    float h = step_size;
    float previous[8];
    for (int i = 0; i < 8; ++i) previous[i] = state[i];
    int final_status = 4;
    float out_intensity = 0.0f;
    float out_redshift = 0.0f;
    float out_temperature = 0.0f;
    float out_null_error = 0.0f;
    for (int step = 1; step <= max_steps; ++step) {
        rk4_step_d(state, a, h);
        float r = state[1];
        float theta = state[2];
        int finite = 1;
        for (int i = 0; i < 8; ++i) { if (!isfinite(state[i])) { finite = 0; break; } }
        if (!finite) {
            int prev_finite = 1;
            for (int i = 0; i < 8; ++i) { if (!isfinite(previous[i])) { prev_finite = 0; break; } }
            float prev_r = previous[1];
            int near_horizon = prev_finite && (prev_r <= r_h + fmaxf(2.0f, 8.0f * fabsf(h)));
            int inward = prev_finite && (previous[5] < 0.0f);
            final_status = (near_horizon && inward) ? 2 : 5;
            if (prev_finite) {
                float gtt, gtphi, grr, gtheta, gphi;
                inverse_metric_terms_d(prev_r, previous[2], a, &gtt, &gtphi, &grr, &gtheta, &gphi);
                float pt = previous[4], pr = previous[5], ptheta = previous[6], pphi = previous[7];
                float ham = 0.5f * (gtt * pt * pt + 2.0f * gtphi * pt * pphi + grr * pr * pr + gtheta * ptheta * ptheta + gphi * pphi * pphi);
                out_null_error = fabsf(ham);
            } else {
                out_null_error = 1.0e6f;
            }
            break;
        }
        if (r <= r_h + horizon_epsilon) {
            final_status = 2;
            float gtt, gtphi, grr, gtheta, gphi;
            inverse_metric_terms_d(r, theta, a, &gtt, &gtphi, &grr, &gtheta, &gphi);
            float ham = 0.5f * (gtt * state[4] * state[4] + 2.0f * gtphi * state[4] * state[7] + grr * state[5] * state[5] + gtheta * state[6] * state[6] + gphi * state[7] * state[7]);
            out_null_error = fabsf(ham);
            break;
        }
        if (r > escape_radius && step > 3) {
            final_status = 3;
            float gtt, gtphi, grr, gtheta, gphi;
            inverse_metric_terms_d(r, theta, a, &gtt, &gtphi, &grr, &gtheta, &gphi);
            float ham = 0.5f * (gtt * state[4] * state[4] + 2.0f * gtphi * state[4] * state[7] + grr * state[5] * state[5] + gtheta * state[6] * state[6] + gphi * state[7] * state[7]);
            out_null_error = fabsf(ham);
            break;
        }
        float prev_theta = previous[2];
        float crossed = (prev_theta - 1.5707963267948966f) * (theta - 1.5707963267948966f);
        if (crossed <= 0.0f && step > 1) {
            float denom = theta - prev_theta;
            float frac = 0.0f;
            if (fabsf(denom) >= 1.0e-12f) frac = (1.5707963267948966f - prev_theta) / denom;
            if (frac < 0.0f) frac = 0.0f;
            if (frac > 1.0f) frac = 1.0f;
            float hit[8];
            for (int i = 0; i < 8; ++i) hit[i] = previous[i] + frac * (state[i] - previous[i]);
            float hit_r = hit[1];
            if (hit_r >= r_in && hit_r <= r_outer) {
                float lambda = hit[7] / fmaxf(1.0e-12f, -hit[4]);
                float intens = hit_intensity_d(hit_r, 1.5707963267948966f, a, lambda, emissivity_q, emission_model, r_in);
                float gfac = redshift_factor_d(hit_r, 1.5707963267948966f, a, lambda);
                if (emission_model == 1) {
                    float rr = fmaxf(hit_r, r_in);
                    out_temperature = powf(fmaxf(powf(rr, -3.0f) * fmaxf(0.0f, 1.0f - sqrtf(r_in / rr)), 0.0f), 0.25f);
                } else {
                    out_temperature = powf(fmaxf(powf(fmaxf(hit_r, 1.0e-6f), -emissivity_q), 0.0f), 0.25f);
                }
                out_intensity = intens;
                out_redshift = gfac;
                final_status = 1;
                float gtt, gtphi, grr, gtheta, gphi;
                inverse_metric_terms_d(hit_r, 1.5707963267948966f, a, &gtt, &gtphi, &grr, &gtheta, &gphi);
                float ham = 0.5f * (gtt * hit[4] * hit[4] + 2.0f * gtphi * hit[4] * hit[7] + grr * hit[5] * hit[5] + gtheta * hit[6] * hit[6] + gphi * hit[7] * hit[7]);
                out_null_error = fabsf(ham);
                break;
            }
        }
        for (int i = 0; i < 8; ++i) previous[i] = state[i];
    }
    if (final_status == 4) {
        float gtt, gtphi, grr, gtheta, gphi;
        inverse_metric_terms_d(state[1], state[2], a, &gtt, &gtphi, &grr, &gtheta, &gphi);
        float ham = 0.5f * (gtt * state[4] * state[4] + 2.0f * gtphi * state[4] * state[7] + grr * state[5] * state[5] + gtheta * state[6] * state[6] + gphi * state[7] * state[7]);
        out_null_error = fabsf(ham);
    }
    intensity[idx] = out_intensity;
    redshift[idx] = out_redshift;
    temperature[idx] = out_temperature;
    hit_mask[idx] = (final_status == 1) ? 1 : 0;
    status_code[idx] = (unsigned char)final_status;
    null_error[idx] = out_null_error;
}

int main(int argc, char** argv) {
    const int res = (argc > 1) ? atoi(argv[1]) : 48;
    const int threads = 16;
    dim3 block(threads, threads);
    dim3 grid((res + threads - 1) / threads, (res + threads - 1) / threads);

    float* d_intensity;
    float* d_redshift;
    float* d_temperature;
    unsigned char* d_hit_mask;
    unsigned char* d_status_code;
    float* d_null_error;

    cudaMalloc(&d_intensity, res * res * sizeof(float));
    cudaMalloc(&d_redshift, res * res * sizeof(float));
    cudaMalloc(&d_temperature, res * res * sizeof(float));
    cudaMalloc(&d_hit_mask, res * res * sizeof(unsigned char));
    cudaMalloc(&d_status_code, res * res * sizeof(unsigned char));
    cudaMalloc(&d_null_error, res * res * sizeof(float));

    kerr_geodesic_kernel<<<grid, block>>>(
        res, res,
        0.7f, 1.0471975512f, 24.0f,
        3.393f, 28.0f, 3.0f,
        0, 60.0f,
        0.35f, 700,
        0.05f, 90.0f,
        d_intensity, d_redshift, d_temperature,
        d_hit_mask, d_status_code, d_null_error
    );
    cudaDeviceSynchronize();

    cudaFree(d_intensity);
    cudaFree(d_redshift);
    cudaFree(d_temperature);
    cudaFree(d_hit_mask);
    cudaFree(d_status_code);
    cudaFree(d_null_error);
    return 0;
}
