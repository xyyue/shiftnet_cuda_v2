/*
Copyright 2017 the shiftnet_cuda_v2 authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <cuda_runtime.h>
#include <stdio.h>
//#include <algorithmic>

#define MAX_BLOCKS 128

//using std::min;

__global__ void shiftnet_cuda_moduloshift3x3_nchw_float32_kernel_tilein16x16_tileout14x14(
    float *src,
    float *dst,
    int num_h_tiles,
    int num_w_tiles,
    int batch_sz,
    int channels,
    int height,
    int width)
{
  __shared__ float cache[256];
  const int num_blocks = batch_sz * channels * num_h_tiles * num_w_tiles;
  const int num_threads = blockDim.x * num_blocks;
  const int rd_chans = (channels / 9) * 9;
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < num_threads; idx += blockDim.x * gridDim.x)
  {
    const int w_tile_idx = (idx / 256) % num_w_tiles;
    const int h_tile_idx = ((idx / 256) / num_w_tiles) % num_h_tiles;
    const int tile_ch = (((idx / 256) / num_w_tiles) / num_h_tiles) % channels;
    const int tile_batch_idx = ((((idx / 256) / num_w_tiles) / num_h_tiles) / channels) % batch_sz;
    const int w_shift = ((tile_ch % 3) - 1) * (tile_ch < rd_chans);
    const int h_shift = (((tile_ch / 3) % 3) - 1) * (tile_ch < rd_chans);
    const int w_tile_off = threadIdx.x % 16;
    const int h_tile_off = threadIdx.x / 16;
    const int w_idx = w_tile_off - 1 + 14 * w_tile_idx;
    const int h_idx = h_tile_off - 1 + 14 * h_tile_idx;
    const int buf_idx = w_idx + width * (h_idx + height * (tile_ch + channels * tile_batch_idx));
    if (w_idx >= 0 && w_idx < width && h_idx >= 0 && h_idx < height) {
      cache[threadIdx.x] = src[buf_idx];
    } else {
      cache[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    if (w_tile_off >= 1 && w_tile_off < 15 && h_tile_off >= 1 && h_tile_off < 15) {
      if (w_idx >= 0 && w_idx < width && h_idx >= 0 && h_idx < height) {
        const int cache_idx = (w_tile_off + w_shift) + 16 * (h_tile_off + h_shift);
        dst[buf_idx] = cache[cache_idx];
      }
    }
    __syncthreads();
  }
}

extern "C" void shiftnet_cuda_moduloshift3x3_nchw_float32(
    float *src,
    float *dst,
    int batch_sz,
    int channels,
    int height,
    int width,
    cudaStream_t stream)
{
  int num_h_tiles = (height + 14 - 1) / 14;
  int num_w_tiles = (width + 14 - 1) / 14;
  int num_blocks = min(MAX_BLOCKS, batch_sz * channels * num_h_tiles * num_w_tiles);
  shiftnet_cuda_moduloshift3x3_nchw_float32_kernel_tilein16x16_tileout14x14<<<num_blocks, 256, 0, stream>>>(
      src, dst, num_h_tiles, num_w_tiles, batch_sz, channels, height, width);
}

__global__ void shiftnet_cuda_moduloshift3x3bwd_nchw_float32_kernel_tilein16x16_tileout14x14(
    float *src,
    float *dst,
    int num_h_tiles,
    int num_w_tiles,
    int batch_sz,
    int channels,
    int height,
    int width)
{
  __shared__ float cache[256];
  const int num_blocks = batch_sz * channels * num_h_tiles * num_w_tiles;
  const int num_threads = blockDim.x * num_blocks;
  const int rd_chans = (channels / 9) * 9;
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < num_threads; idx += blockDim.x * gridDim.x)
  {
    const int w_tile_idx = (idx / 256) % num_w_tiles;
    const int h_tile_idx = ((idx / 256) / num_w_tiles) % num_h_tiles;
    const int tile_ch = (((idx / 256) / num_w_tiles) / num_h_tiles) % channels;
    const int tile_batch_idx = ((((idx / 256) / num_w_tiles) / num_h_tiles) / channels) % batch_sz;
    const int w_shift = (1 - (tile_ch % 3)) * (tile_ch < rd_chans);
    const int h_shift = (1 - ((tile_ch / 3) % 3)) * (tile_ch < rd_chans);
    const int w_tile_off = threadIdx.x % 16;
    const int h_tile_off = threadIdx.x / 16;
    const int w_idx = w_tile_off - 1 + 14 * w_tile_idx;
    const int h_idx = h_tile_off - 1 + 14 * h_tile_idx;
    const int buf_idx = w_idx + width * (h_idx + height * (tile_ch + channels * tile_batch_idx));
    if (w_idx >= 0 && w_idx < width && h_idx >= 0 && h_idx < height) {
      cache[threadIdx.x] = src[buf_idx];
    } else {
      cache[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    if (w_tile_off >= 1 && w_tile_off < 15 && h_tile_off >= 1 && h_tile_off < 15) {
      if (w_idx >= 0 && w_idx < width && h_idx >= 0 && h_idx < height) {
        const int cache_idx = (w_tile_off + w_shift) + 16 * (h_tile_off + h_shift);
        dst[buf_idx] = cache[cache_idx];
      }
    }
    __syncthreads();
  }
}

extern "C" void shiftnet_cuda_moduloshift3x3bwd_nchw_float32(
    float *src,
    float *dst,
    int batch_sz,
    int channels,
    int height,
    int width,
    cudaStream_t stream)
{
  int num_h_tiles = (height + 14 - 1) / 14;
  int num_w_tiles = (width + 14 - 1) / 14;
  int num_blocks = min(MAX_BLOCKS, batch_sz * channels * num_h_tiles * num_w_tiles);
  shiftnet_cuda_moduloshift3x3bwd_nchw_float32_kernel_tilein16x16_tileout14x14<<<num_blocks, 256, 0, stream>>>(
      src, dst, num_h_tiles, num_w_tiles, batch_sz, channels, height, width);
}

__global__ void shiftnet_cuda_moduloshiftgeneric_nchw_float32_kernel_tilein16x16(
    float *src,
    float *dst,
    int num_h_tiles,
    int num_w_tiles,
    int batch_sz,
    int channels,
    int height,
    int width,
    int kernel_size,
    int dilate_factor,
    int direction)
{
  __shared__ float cache[256];
  const int num_blocks = batch_sz * channels * num_h_tiles * num_w_tiles;
  const int num_threads = blockDim.x * num_blocks;
  const int rd_chans = (channels / (kernel_size * kernel_size)) * (kernel_size * kernel_size);
  const int half_kernel_size = kernel_size / 2;
  const int dilated_half_kernel_size = dilate_factor * half_kernel_size;
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
      idx < num_threads; idx += blockDim.x * gridDim.x)
  {
    const int w_tile_idx = (idx / 256) % num_w_tiles;
    const int h_tile_idx = ((idx / 256) / num_w_tiles) % num_h_tiles;
    const int tile_ch = (((idx / 256) / num_w_tiles) / num_h_tiles) % channels;
    const int tile_batch_idx = ((((idx / 256) / num_w_tiles) / num_h_tiles) / channels) % batch_sz;
    const int w_shift = direction * dilate_factor * (tile_ch < rd_chans) * ((tile_ch % kernel_size) - half_kernel_size);
    const int h_shift = direction * dilate_factor * (tile_ch < rd_chans) * (((tile_ch / kernel_size) % kernel_size) - half_kernel_size);
    const int w_tile_off = threadIdx.x % 16;
    const int h_tile_off = threadIdx.x / 16;
    const int w_idx = w_tile_off - dilated_half_kernel_size + (16 - 2 * dilated_half_kernel_size) * w_tile_idx;
    const int h_idx = h_tile_off - dilated_half_kernel_size + (16 - 2 * dilated_half_kernel_size) * h_tile_idx;
    const int buf_idx = w_idx + width * (h_idx + height * (tile_ch + channels * tile_batch_idx));
    if (w_idx >= 0 && w_idx < width && h_idx >= 0 && h_idx < height) {
      cache[threadIdx.x] = src[buf_idx];
    } else {
      cache[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    if (w_tile_off >= dilated_half_kernel_size &&
        w_tile_off <  (16 - dilated_half_kernel_size) &&
        h_tile_off >= dilated_half_kernel_size &&
        h_tile_off <  (16 - dilated_half_kernel_size))
    {
      if (w_idx >= 0 && w_idx < width && h_idx >= 0 && h_idx < height) {
        const int cache_idx = (w_tile_off + w_shift) + 16 * (h_tile_off + h_shift);
        dst[buf_idx] = cache[cache_idx];
      }
    }
    __syncthreads();
  }
}

__global__ void shiftnet_cuda_shift_generic_nchw_float32_kernel_tilein16x16_ctrl(
    float *src,
    float *dst,
    float *ctrl,
    int num_h_tiles,
    int num_w_tiles,
    int batch_sz,
    int channels,
    int height,
    int width,
    int kernel_size,
    int dilate_factor,
    int direction)
{
  __shared__ float cache[256];
  // __shared__ int ctrl_val;
  const int num_blocks = batch_sz * channels * num_h_tiles * num_w_tiles;
  const int num_threads = blockDim.x * num_blocks;
  // const int rd_chans = (channels / (kernel_size * kernel_size)) * (kernel_size * kernel_size);
  const int half_kernel_size = kernel_size / 2;
  const int dilated_half_kernel_size = dilate_factor * half_kernel_size;
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < num_threads; idx += blockDim.x * gridDim.x)
  {
    const int w_tile_idx = (idx / 256) % num_w_tiles;
    const int h_tile_idx = ((idx / 256) / num_w_tiles) % num_h_tiles;
    const int tile_ch = (((idx / 256) / num_w_tiles) / num_h_tiles) % channels;
    const int tile_batch_idx = ((((idx / 256) / num_w_tiles) / num_h_tiles) / channels) % batch_sz;
    // const int w_shift = direction * dilate_factor * (tile_ch < rd_chans) * ((tile_ch % kernel_size) - half_kernel_size);
    // const int h_shift = direction * dilate_factor * (tile_ch < rd_chans) * (((tile_ch / kernel_size) % kernel_size) - half_kernel_size);
    // if (idx % 256 == 0)
    //   ctrl_val = ctrl[tile_ch + channels * tile_batch_idx];
    // __syncthreads();
    // int l_r = (ctrl_val - 1) % 3 - 1;
    // int u_d = (ctrl_val - 1) / 3 - 1;

    // const int w_shift = direction * dilate_factor * l_r;
    // const int h_shift = direction * dilate_factor * u_d;


    const int w_tile_off = threadIdx.x % 16;
    const int h_tile_off = threadIdx.x / 16;
    const int w_idx = w_tile_off - dilated_half_kernel_size + (16 - 2 * dilated_half_kernel_size) * w_tile_idx;
    const int h_idx = h_tile_off - dilated_half_kernel_size + (16 - 2 * dilated_half_kernel_size) * h_tile_idx;

    const int w_shift = direction * ctrl[0 + 2 * (w_idx + width * (h_idx + height * (tile_ch + channels * tile_batch_idx)))];
    const int h_shift = direction * ctrl[1 + 2 * (w_idx + width * (h_idx + height * (tile_ch + channels * tile_batch_idx)))];


    const int buf_idx = w_idx + width * (h_idx + height * (tile_ch + channels * tile_batch_idx));
    if (w_idx >= 0 && w_idx < width && h_idx >= 0 && h_idx < height) {
      cache[threadIdx.x] = src[buf_idx];
    } else {
      cache[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    if (w_tile_off >= dilated_half_kernel_size &&
        w_tile_off <  (16 - dilated_half_kernel_size) &&
        h_tile_off >= dilated_half_kernel_size &&
        h_tile_off <  (16 - dilated_half_kernel_size))
    {
      if (w_idx >= 0 && w_idx < width && h_idx >= 0 && h_idx < height) {
        const int cache_idx = (w_tile_off + w_shift) + 16 * (h_tile_off + h_shift);
        // printf("w_tile_off: %d, w_shift: %d, h_tile_off: %d, h_shift: %d, Idx : %d \n", w_tile_off, w_shift, h_tile_off, h_shift, cache_idx);
        dst[buf_idx] = cache[cache_idx];
      }
    }
    __syncthreads();
  }
}

__global__ void shiftnet_cuda_shift_generic_nchw_float32_kernel_tilein16x16_ctrl2D(
    float *src,
    float *dst,
    float *ctrl,
    float *ctrl_grad,
    int num_h_tiles,
    int num_w_tiles,
    int batch_sz,
    int channels,
    int height,
    int width,
    int kernel_size,
    int dilate_factor,
    int direction)
{
  __shared__ float cache[256];
  // __shared__ int ctrl_val;
  const int num_blocks = batch_sz * channels * num_h_tiles * num_w_tiles;
  const int num_threads = blockDim.x * num_blocks;
  // const int rd_chans = (channels / (kernel_size * kernel_size)) * (kernel_size * kernel_size);
  const int half_kernel_size = kernel_size / 2;
  const int dilated_half_kernel_size = dilate_factor * half_kernel_size;
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < num_threads; idx += blockDim.x * gridDim.x)
  {
    const int w_tile_idx = (idx / 256) % num_w_tiles;
    const int h_tile_idx = ((idx / 256) / num_w_tiles) % num_h_tiles;
    const int tile_ch = (((idx / 256) / num_w_tiles) / num_h_tiles) % channels;
    const int tile_batch_idx = ((((idx / 256) / num_w_tiles) / num_h_tiles) / channels) % batch_sz;
    // const int w_shift = direction * dilate_factor * (tile_ch < rd_chans) * ((tile_ch % kernel_size) - half_kernel_size);
    // const int h_shift = direction * dilate_factor * (tile_ch < rd_chans) * (((tile_ch / kernel_size) % kernel_size) - half_kernel_size);
    // if (idx % 256 == 0)
    //   ctrl_val = ctrl[tile_ch + channels * tile_batch_idx];
    // __syncthreads();
    // int l_r = (ctrl_val - 1) % 3 - 1;
    // int u_d = (ctrl_val - 1) / 3 - 1;

    // const int w_shift = direction * dilate_factor * l_r;
    // const int h_shift = direction * dilate_factor * u_d;


    const int w_tile_off = threadIdx.x % 16;
    const int h_tile_off = threadIdx.x / 16;
    const int w_idx = w_tile_off - dilated_half_kernel_size + (16 - 2 * dilated_half_kernel_size) * w_tile_idx;
    const int h_idx = h_tile_off - dilated_half_kernel_size + (16 - 2 * dilated_half_kernel_size) * h_tile_idx;

    const int w_shift = direction * ctrl[0 + 2 * (w_idx + width * (h_idx + height * (tile_ch + channels * tile_batch_idx)))];
    const int h_shift = direction * ctrl[1 + 2 * (w_idx + width * (h_idx + height * (tile_ch + channels * tile_batch_idx)))];


    const int buf_idx = w_idx + width * (h_idx + height * (tile_ch + channels * tile_batch_idx));
    if (w_idx >= 0 && w_idx < width && h_idx >= 0 && h_idx < height) {
      cache[threadIdx.x] = src[buf_idx];
    } else {
      cache[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    if (w_tile_off >= dilated_half_kernel_size &&
        w_tile_off <  (16 - dilated_half_kernel_size) &&
        h_tile_off >= dilated_half_kernel_size &&
        h_tile_off <  (16 - dilated_half_kernel_size))
    {
      if (w_idx >= 0 && w_idx < width && h_idx >= 0 && h_idx < height) {
        const int cache_idx = (w_tile_off + w_shift) + 16 * (h_tile_off + h_shift);
        dst[buf_idx] = cache[cache_idx];
        if (direction == -1) {
          const int h_in_kernel = -1 * h_shift + kernel_size / 2;
          const int w_in_kernel = -1 * w_shift + kernel_size / 2;
          const int grad_idx = w_in_kernel + h_in_kernel * kernel_size + kernel_size * kernel_size * (w_idx + width * (h_idx + height * (tile_ch + channels * tile_batch_idx)));
          const int cache_grad_idx = (w_tile_off - w_shift) + 16 * (h_tile_off - h_shift);
          ctrl_grad[grad_idx] = cache[cache_grad_idx];
          // printf("w_idx: %d, h_idx: %d, h_in_kernel: %d, w_in_kernel: %d, w_shift: %d, h_shift: %d, val: %d\n", w_idx, h_idx, h_in_kernel, w_in_kernel, w_shift, h_shift, (int)cache[cache_grad_idx]);
        }
      }


    }
    __syncthreads();
  }
}

extern "C" void shiftnet_cuda_moduloshiftgeneric_nchw_float32(
    float *src,
    float *dst,
    int batch_sz,
    int channels,
    int height,
    int width,
    int kernel_size,
    int dilate_factor,
    int direction,
    cudaStream_t stream)
{
  int dilated_half_kernel_size = dilate_factor * (kernel_size / 2);
  int tile_out_size = 16 - 2 * dilated_half_kernel_size;
  int num_h_tiles = (height + tile_out_size - 1) / tile_out_size;
  int num_w_tiles = (width + tile_out_size - 1) / tile_out_size;
  int num_blocks = min(MAX_BLOCKS, batch_sz * channels * num_h_tiles * num_w_tiles);
  shiftnet_cuda_moduloshiftgeneric_nchw_float32_kernel_tilein16x16<<<num_blocks, 256, 0, stream>>>(
      src, dst, num_h_tiles, num_w_tiles, batch_sz, channels, height, width, kernel_size, dilate_factor, direction);
}


extern "C" void shiftnet_cuda_shift_generic_nchw_float32_ctrl(
    float *src,
    float *dst,
    float *ctrl,
    int batch_sz,
    int channels,
    int height,
    int width,
    int kernel_size,
    int dilate_factor,
    int direction,
    cudaStream_t stream)
{
  int dilated_half_kernel_size = dilate_factor * (kernel_size / 2);
  int tile_out_size = 16 - 2 * dilated_half_kernel_size;
  int num_h_tiles = (height + tile_out_size - 1) / tile_out_size;
  int num_w_tiles = (width + tile_out_size - 1) / tile_out_size;
  int num_blocks = min(MAX_BLOCKS, batch_sz * channels * num_h_tiles * num_w_tiles);
  shiftnet_cuda_shift_generic_nchw_float32_kernel_tilein16x16_ctrl<<<num_blocks, 256, 0, stream>>>(
      src, dst, ctrl, num_h_tiles, num_w_tiles, batch_sz, channels, height, width, kernel_size, dilate_factor, direction);
}

extern "C" void shiftnet_cuda_shift_generic_nchw_float32_ctrl2D(
    float *src,
    float *dst,
    float *ctrl,
    float *ctrl_grad,
    int batch_sz,
    int channels,
    int height,
    int width,
    int kernel_size,
    int dilate_factor,
    int direction,
    cudaStream_t stream)
{
  int dilated_half_kernel_size = dilate_factor * (kernel_size / 2);
  int tile_out_size = 16 - 2 * dilated_half_kernel_size;
  int num_h_tiles = (height + tile_out_size - 1) / tile_out_size;
  int num_w_tiles = (width + tile_out_size - 1) / tile_out_size;
  int num_blocks = min(MAX_BLOCKS, batch_sz * channels * num_h_tiles * num_w_tiles);
  shiftnet_cuda_shift_generic_nchw_float32_kernel_tilein16x16_ctrl2D<<<num_blocks, 256, 0, stream>>>(
      src, dst, ctrl, ctrl_grad, num_h_tiles, num_w_tiles, batch_sz, channels, height, width, kernel_size, dilate_factor, direction);
}
