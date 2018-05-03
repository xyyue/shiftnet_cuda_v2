import sys
sys.path.append("./")

import shiftnet_cuda

import numpy as np
import torch
import torch.cuda

num = 6 
n_batch = 1
n_channel = 9
def main():
  pattern = np.arange(num * num).reshape(num, num)
  src_buf = np.zeros((n_batch, n_channel, num, num)).astype(np.float32)
  for bnr in range(n_batch):
    for ch in range(n_channel):
      src_buf[bnr,ch,:,:] = pattern

  x_hin = torch.zeros(n_batch, n_channel, num, num).type(torch.FloatTensor)
  #x_hin[:,:,1:4,1:4] = 1.0
  x_hin.copy_(torch.from_numpy(src_buf))

  y_hin = torch.zeros(n_batch, n_channel, num, num).type(torch.FloatTensor)

  x = x_hin.cuda()
  y = y_hin.cuda()

  ctrl_cpu = torch.ones(n_batch, n_channel, num, num, 2)
  ctrl_cpu[0][0][0][0][0] = 2
  ctrl_cpu[0][0][0][0][1] = 2
  ctrl_cpu[0][0][4][4][0] = -2
  ctrl_cpu[0][0][4][4][1] = -1
  # ctrl_cpu = ctrl_cpu.type(torch.IntTensor)
  ctrl = ctrl_cpu.cuda()
  #ret = shiftnet_cuda.moduloshift3x3_nchw(x, y)
  ret = shiftnet_cuda.moduloshiftgeneric_nchw(x, y, ctrl, 3, 1, 1)
  assert ret == 1

  x_hout = x.cpu()
  y_hout = y.cpu()

  print(x_hout[0,0,:num,:num])
  for ch in range(9):
    print(y_hout[0,ch,:num,:num])

if __name__ == "__main__":
  main()
