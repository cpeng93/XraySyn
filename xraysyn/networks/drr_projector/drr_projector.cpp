#include <torch/extension.h>

#include <vector>

// CUDA forward/backward declarations

torch::Tensor dp_nearest_cuda_forward(
    torch::Tensor volume,
    torch::Tensor detector_shape,
    torch::Tensor ray_mat,
    torch::Tensor source,
    torch::Tensor step_size,
    torch::Tensor voxel_size);

torch::Tensor dp_nearest_cuda_backward(
  torch::Tensor projection,
  torch::Tensor volume_shape,
  torch::Tensor ray_mat,
  torch::Tensor source,
  torch::Tensor step_size,
  torch::Tensor voxel_size);

torch::Tensor dp_trilinear_cuda_forward(
    torch::Tensor volume,
    torch::Tensor detector_shape,
    torch::Tensor ray_mat,
    torch::Tensor source,
    torch::Tensor step_size,
    torch::Tensor voxel_size);

torch::Tensor dp_trilinear_cuda_backward(
  torch::Tensor projection,
  torch::Tensor volume_shape,
  torch::Tensor ray_mat,
  torch::Tensor source,
  torch::Tensor step_size,
  torch::Tensor voxel_size);

torch::Tensor dp_backproject_nearest_cuda_forward(
  torch::Tensor projection,
  torch::Tensor volume_shape,
  torch::Tensor ray_mat,
  torch::Tensor source,
  torch::Tensor step_size,
  torch::Tensor voxel_size);

torch::Tensor dp_backproject_trilinear_cuda_forward(
  torch::Tensor projection,
  torch::Tensor volume_shape,
  torch::Tensor ray_mat,
  torch::Tensor source,
  torch::Tensor step_size,
  torch::Tensor voxel_size);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor dp_nearest_forward(
    torch::Tensor volume,
    torch::Tensor detector_shape,
    torch::Tensor ray_mat,
    torch::Tensor source,
    torch::Tensor step_size,
    torch::Tensor voxel_size) {
  CHECK_INPUT(volume);
  CHECK_INPUT(ray_mat);
  CHECK_INPUT(source);

  return dp_nearest_cuda_forward(volume, detector_shape, ray_mat, source, step_size, voxel_size);
}

torch::Tensor dp_trilinear_forward(
    torch::Tensor volume,
    torch::Tensor detector_shape,
    torch::Tensor ray_mat,
    torch::Tensor source,
    torch::Tensor step_size,
    torch::Tensor voxel_size) {
  CHECK_INPUT(volume);
  CHECK_INPUT(ray_mat);
  CHECK_INPUT(source);

  return dp_trilinear_cuda_forward(volume, detector_shape, ray_mat, source, step_size, voxel_size);
}

torch::Tensor dp_nearest_backward(
    torch::Tensor projection,
    torch::Tensor volume_shape,
    torch::Tensor ray_mat,
    torch::Tensor source,
    torch::Tensor step_size,
    torch::Tensor voxel_size) {
  CHECK_INPUT(projection);
  CHECK_INPUT(ray_mat);
  CHECK_INPUT(source);

  return dp_nearest_cuda_backward(projection, volume_shape, ray_mat, source, step_size, voxel_size);
}




torch::Tensor dp_trilinear_backward(
    torch::Tensor projection,
    torch::Tensor volume_shape,
    torch::Tensor ray_mat,
    torch::Tensor source,
    torch::Tensor step_size,
    torch::Tensor voxel_size) {
  CHECK_INPUT(projection);
  CHECK_INPUT(ray_mat);
  CHECK_INPUT(source);

  return dp_trilinear_cuda_backward(projection, volume_shape, ray_mat, source, step_size, voxel_size);
}



torch::Tensor dp_backproject_nearest_forward(
    torch::Tensor projection,
    torch::Tensor volume_shape,
    torch::Tensor ray_mat,
    torch::Tensor source,
    torch::Tensor step_size,
    torch::Tensor voxel_size) {
  CHECK_INPUT(projection);
  CHECK_INPUT(ray_mat);
  CHECK_INPUT(source);

  return dp_backproject_nearest_cuda_forward(projection, volume_shape, ray_mat, source, step_size, voxel_size);
}

torch::Tensor dp_backproject_trilinear_forward(
    torch::Tensor projection,
    torch::Tensor volume_shape,
    torch::Tensor ray_mat,
    torch::Tensor source,
    torch::Tensor step_size,
    torch::Tensor voxel_size) {
  CHECK_INPUT(projection);
  CHECK_INPUT(ray_mat);
  CHECK_INPUT(source);

  return dp_backproject_trilinear_cuda_forward(projection, volume_shape, ray_mat, source, step_size, voxel_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nearest_forward", &dp_nearest_forward, "drr projector nearest forward (CUDA)");
  m.def("nearest_backward", &dp_nearest_backward, "drr projector nearest backward (CUDA)");
  m.def("trilinear_forward", &dp_trilinear_forward, "drr projector trilinear forward (CUDA)");
  m.def("trilinear_backward", &dp_trilinear_backward, "drr projector trilinear backward (CUDA)");
  m.def("backproject_nearest_forward", &dp_backproject_nearest_forward, "drr projector backproject nearest forward (CUDA)");
  m.def("backproject_trilinear_forward", &dp_backproject_nearest_forward, "drr projector backproject nearest forward (CUDA)");
}
