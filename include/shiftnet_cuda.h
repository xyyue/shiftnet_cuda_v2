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

int moduloshift3x3_nchw(THCudaTensor *src_tensor, THCudaTensor *dst_tensor);
int moduloshift3x3bwd_nchw(THCudaTensor *src_tensor, THCudaTensor *dst_tensor);
int moduloshiftgeneric_nchw(THCudaTensor *src_tensor, THCudaTensor *dst_tensor, int kernel_size, int dilate_factor, int direction);
int shift_generic_nchw_ctrl(THCudaTensor *src_tensor, THCudaTensor *dst_tensor, THCudaTensor *ctrl_tensor, int kernel_size, int dilate_factor, int direction);
int shift_generic_nchw_ctrl_grad(THCudaTensor *src_tensor, THCudaTensor *dst_tensor, THCudaTensor *ctrl_tensor, THCudaTensor *ctrl_grad_tensor, int kernel_size, int dilate_factor, int direction);
