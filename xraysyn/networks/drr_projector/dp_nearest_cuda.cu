#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <vector>

using namespace std;
using scalar_t = float;

#include <stdio.h>


const int MAX_BATCH_SIZE = 64;
__constant__ scalar_t RAY_MAT[9 * MAX_BATCH_SIZE];
__constant__ scalar_t RAY[3 * MAX_BATCH_SIZE];


__global__ void dp_nearest_cuda_forward_kernel(
  scalar_t* __restrict__ volume,
  scalar_t* __restrict__ projection,
  int nx, int ny, int nz,
  int dh, int dw, int bs,
  scalar_t step, scalar_t vx, scalar_t vy, scalar_t vz
) {
  //xray coordinate calculation
  int udx = threadIdx.x + blockIdx.x * blockDim.x;
  int vdx = threadIdx.y + blockIdx.y * blockDim.y;
  // batch memory location
  int idx = blockIdx.z * dw * dh + udx * dw + vdx;
  // if (idx == 14968){
    // printf("projection index %d, nx %d, ny %d, nz %d\n", idx, nx, ny, nz);
  // }

  if (udx >= dh || vdx >= dw) { return; }

  //depends on batch size, find the pointer to raymat and ray, raymat is the transformation knowledge about the projected plane
  scalar_t* ray_mat = RAY_MAT + (blockIdx.z % bs) * 9;
  scalar_t* ray = RAY + (blockIdx.z % bs) * 3;
  // u,v are coordinates on xray, projected by ray mat to get 3d coorinates sx sy sz
  scalar_t u = (scalar_t) udx + 0.5f;
  scalar_t v = (scalar_t) vdx + 0.5f;
  //ray light source location
  scalar_t sx = ray[0];
  scalar_t sy = ray[1];
  scalar_t sz = ray[2];

  // compute ray direction
  scalar_t rx = ray_mat[2] + v * ray_mat[1] + u * ray_mat[0];
  scalar_t ry = ray_mat[5] + v * ray_mat[4] + u * ray_mat[3];
  scalar_t rz = ray_mat[8] + v * ray_mat[7] + u * ray_mat[6];

  // normalize ray direction
  scalar_t nf = 1.0f / (sqrt((rx * rx) + (ry * ry) + (rz * rz)));
  rx *= nf;
  ry *= nf;
  rz *= nf;

  //calculate projections
  // Step 1: compute alpha value at entry and exit point of the volume, this is to optimzie such that for air return 0
  //calculates how long the ray are in the volume at what location
  scalar_t minAlpha, maxAlpha;
  scalar_t alpha0, alpha1;

  minAlpha = 0.0f;
  maxAlpha = INFINITY;

  if (0.0f != rx)
  {
    alpha0 = -sx / rx;
    alpha1 = (nx - sx) / rx;
    minAlpha = fmin(alpha0, alpha1);
    maxAlpha = fmax(alpha0, alpha1);
  } else if (0.0f > sx || sx > nx) {
    return;
  }

  if (0.0f != ry)
  {
    alpha0 = -sy / ry;
    alpha1 = (ny - sy) / ry;
    minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
    maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
  } else if (0.0f > sy || sy > ny) {
    return;
  }

  if (0.0f != rz)
  {
    alpha0 = - sz / rz;
    alpha1 = (nz - sz) / rz;
    minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
    maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
  } else if (0.0f > sz || sz > nz) {
    return;
  }

  // Step 2: Cast ray if it intersects the volume

  // Trapezoidal rule (interpolating function = piecewise linear func)
  scalar_t px, py, pz;
  int ix, iy, iz;
  int nyz = ny * nz;

  sx = sx - 0.5f;
  sy = sy - 0.5f;
  sz = sz - 0.5f;
  volume = volume + nx * nyz * blockIdx.z;

  // Entrance boundary
  // In CUDA, voxel centers are located at (xx.5, xx.5, xx.5),
  // whereas, SwVolume has voxel centers at integers.
  // For the initial interpolated value, only a half stepsize is
  //  considered in the computation.
  if (minAlpha < maxAlpha) {
    px = sx + minAlpha * rx;
    py = sy + minAlpha * ry;
    pz = sz + minAlpha * rz;

    ix = lroundf(px);
    iy = lroundf(py);
    iz = lroundf(pz);
    //multiple by 0.5 becuase the first point is half in volume half out volume
    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
      // if (idx == 14968){
      //   printf("first project value %f, ix %d, iy %d, iz %d, minAlpha %f.\n", 0.5f * step * volume[nyz * ix + nz * iy + iz], ix, iy, iz, minAlpha);
      // }
      projection[idx] += 0.5f * volume[nyz * ix + nz * iy + iz];
    }
    minAlpha += step;
  }

  // Mid segments
  while (minAlpha < maxAlpha)
  {
    px = sx + minAlpha * rx;
    py = sy + minAlpha * ry;
    pz = sz + minAlpha * rz;

    ix = lroundf(px);
    iy = lroundf(py);
    iz = lroundf(pz);
    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
      // if (idx == 14968){
      //   printf("project value %f, ix %d, iy %d, iz %d, minAlpha %f.\n", step * volume[nyz * ix + nz * iy + iz], ix, iy, iz, minAlpha);
      // }
      projection[idx] += volume[nyz * ix + nz * iy + iz];
    }
    minAlpha += step;
  }
  // if (idx == 14968){
  //   printf("forward projection value at mid %f\n", projection[idx]);
  // }
  // Scaling by stepsize;
  projection[idx] *= step;

  if (projection[idx] > 0.0f ) {
    minAlpha -= step;
    scalar_t lastStepsize = maxAlpha - minAlpha;
    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
      // if (idx == 14968){
      //   printf("last project value1 %f, ix %d, iy %d, iz %d, minAlpha %f.\n", 0.5f* lastStepsize * volume[nyz * ix + nz * iy + iz], ix, iy, iz, minAlpha);
      // }
      projection[idx] -= 0.5f * step * volume[nyz * ix + nz * iy + iz];
      projection[idx] += 0.5f * lastStepsize * volume[nyz * ix + nz * iy + iz];
    }
    px = sx + maxAlpha * rx;
    py = sy + maxAlpha * ry;
    pz = sz + maxAlpha * rz;

    // The last segment of the line integral takes care of the
    // varying length.
    ix = lroundf(px);
    iy = lroundf(py);
    iz = lroundf(pz);
    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
      // if (idx == 14968){
      //   printf("last project value2 %f, ix %d, iy %d, iz %d, minAlpha %f.\n", 0.5f * lastStepsize * volume[nyz * ix + nz * iy + iz], ix, iy, iz, minAlpha);
      // }
      projection[idx] += 0.5f * lastStepsize * volume[nyz * ix + nz * iy + iz];
    }
  }
  // if (idx == 14968){
  //   printf("forward projection value %f\n", projection[idx]);
  // }
  projection[idx] *= sqrt((rx * vx)*(rx * vx) + (ry * vy)*(ry * vy) + (rz * vz)*(rz * vz));
  // if (idx == 14968){
  //   printf("forward projection value modified %f\n", projection[idx]);
  // }
}

__global__ void dp_nearest_cuda_backward_kernel(
  scalar_t* __restrict__ volume,
  scalar_t* __restrict__ projection,
  int nx, int ny, int nz,
  int dh, int dw, int bs,
  scalar_t step, scalar_t vx, scalar_t vy, scalar_t vz
) {
  int udx = threadIdx.x + blockIdx.x * blockDim.x;
  int vdx = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = blockIdx.z * dw * dh + udx * dw + vdx;

  if (udx >= dh || vdx >= dw) { return; }

  scalar_t* ray_mat = RAY_MAT + (blockIdx.z % bs) * 9;
  scalar_t* ray = RAY + (blockIdx.z % bs) * 3;
  scalar_t u = (scalar_t) udx + 0.5f;
  scalar_t v = (scalar_t) vdx + 0.5f;
  scalar_t sx = ray[0];
  scalar_t sy = ray[1];
  scalar_t sz = ray[2];

  // compute ray direction
  scalar_t rx = ray_mat[2] + v * ray_mat[1] + u * ray_mat[0];
  scalar_t ry = ray_mat[5] + v * ray_mat[4] + u * ray_mat[3];
  scalar_t rz = ray_mat[8] + v * ray_mat[7] + u * ray_mat[6];

  // normalize ray direction
  scalar_t nf = 1.0f / (sqrt((rx * rx) + (ry * ry) + (rz * rz)));
  rx *= nf;
  ry *= nf;
  rz *= nf;

  //calculate projections
  // Step 1: compute alpha value at entry and exit point of the volume
  scalar_t minAlpha, maxAlpha;
  scalar_t alpha0, alpha1;

  minAlpha = 0.0f;
  maxAlpha = INFINITY;

  if (0.0f != rx)
  {
    alpha0 = -sx / rx;
    alpha1 = (nx - sx) / rx;
    minAlpha = fmin(alpha0, alpha1);
    maxAlpha = fmax(alpha0, alpha1);
  } else if (0.0f > sx || sx > nx) {
    return;
  }

  if (0.0f != ry)
  {
    alpha0 = -sy / ry;
    alpha1 = (ny - sy) / ry;
    minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
    maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
  } else if (0.0f > sy || sy > ny) {
    return;
  }

  if (0.0f != rz)
  {
    alpha0 = - sz / rz;
    alpha1 = (nz - sz) / rz;
    minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
    maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
  } else if (0.0f > sz || sz > nz) {
    return;
  }

  // Step 2: Cast ray if it intersects the volume

  // Trapezoidal rule (interpolating function = piecewise linear func)
  scalar_t px, py, pz;
  scalar_t proj;
  int ix, iy, iz;
  int nyz = ny * nz;

  sx = sx - 0.5f;
  sy = sy - 0.5f;
  sz = sz - 0.5f;
  volume = volume + nx * nyz * blockIdx.z;
  //dL/dV = dL/dp * dp/dV = dL/dp because projection += voxel value with factor of 1 (or the multiplicative factor at the lastline of forward pojrection)
  //projectio here is the gradient of projection
  proj = projection[idx] * step * sqrt((rx * vx)*(rx * vx) + (ry * vy)*(ry * vy) + (rz * vz)*(rz * vz));


  // Entrance boundary
  // In CUDA, voxel centers are located at (xx.5, xx.5, xx.5),
  // whereas, SwVolume has voxel centers at integers.
  // For the initial interpolated value, only a half stepsize is
  //  considered in the computation.
  bool check = minAlpha < maxAlpha;
  if (minAlpha < maxAlpha) {
    px = sx + minAlpha * rx;
    py = sy + minAlpha * ry;
    pz = sz + minAlpha * rz;

    ix = lroundf(px);
    iy = lroundf(py);
    iz = lroundf(pz);
    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
      // projection[idx] += 0.5f * volume[nyz * ix + nz * iy + iz];
      // pointer operation, add vluae of projection to the index of nyz * ix + nz * iy + iz of volume, understand that volume is a pointer
      atomicAdd(volume + (nyz * ix + nz * iy + iz), 0.5 * proj);
    }
    minAlpha += step;
  }

  // Mid segments
  while (minAlpha < maxAlpha)
  {
    px = sx + minAlpha * rx;
    py = sy + minAlpha * ry;
    pz = sz + minAlpha * rz;

    ix = lroundf(px);
    iy = lroundf(py);
    iz = lroundf(pz);
    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
      // projection[idx] += volume[nyz * ix + nz * iy + iz];
      atomicAdd(volume + (nyz * ix + nz * iy + iz), proj);
    }
    minAlpha += step;
  }

  if (check) {
    minAlpha -= step;
    scalar_t c = (maxAlpha - minAlpha) / step;
    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
      // projection[idx] -= 0.5f * step * volume[nyz * ix + nz * iy + iz];
      // projection[idx] += 0.5f * lastStepsize * volume[nyz * ix + nz * iy + iz];
      atomicAdd(volume + (nyz * ix + nz * iy + iz), 0.5f * (c - 1.0f) * proj);
    }
    px = sx + maxAlpha * rx;
    py = sy + maxAlpha * ry;
    pz = sz + maxAlpha * rz;

    // The last segment of the line integral takes care of the
    // varying length.
    ix = lroundf(px);
    iy = lroundf(py);
    iz = lroundf(pz);
    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
      // projection[idx] += 0.5f * lastStepsize * volume[nyz * ix + nz * iy + iz];
      atomicAdd(volume + (nyz * ix + nz * iy + iz), 0.5f * c * proj);
    }
  }
}


__global__ void dp_backproject_nearest_cuda_forward_kernel(
  scalar_t* __restrict__ volume,
  scalar_t* __restrict__ projection,
  int nx, int ny, int nz,
  int dh, int dw, int bs,
  scalar_t step, scalar_t vx, scalar_t vy, scalar_t vz
) {
  int udx = threadIdx.x + blockIdx.x * blockDim.x;
  int vdx = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = blockIdx.z * dw * dh + udx * dw + vdx;

  if (udx >= dh || vdx >= dw) { return; }

  scalar_t* ray_mat = RAY_MAT + (blockIdx.z % bs) * 9;
  scalar_t* ray = RAY + (blockIdx.z % bs) * 3;
  scalar_t u = (scalar_t) udx + 0.5f;
  scalar_t v = (scalar_t) vdx + 0.5f;
  scalar_t sx = ray[0];
  scalar_t sy = ray[1];
  scalar_t sz = ray[2];

  // compute ray direction
  scalar_t rx = ray_mat[2] + v * ray_mat[1] + u * ray_mat[0];
  scalar_t ry = ray_mat[5] + v * ray_mat[4] + u * ray_mat[3];
  scalar_t rz = ray_mat[8] + v * ray_mat[7] + u * ray_mat[6];

  // normalize ray direction
  scalar_t nf = 1.0f / (sqrt((rx * rx) + (ry * ry) + (rz * rz)));
  rx *= nf;
  ry *= nf;
  rz *= nf;



  //calculate projections
  // Step 1: compute alpha value at entry and exit point of the volume
  scalar_t minAlpha, maxAlpha;
  scalar_t alpha0, alpha1;

  minAlpha = 0.0f;
  maxAlpha = INFINITY;

  if (0.0f != rx)
  {
    alpha0 = -sx / rx;
    alpha1 = (nx - sx) / rx;
    minAlpha = fmin(alpha0, alpha1);
    maxAlpha = fmax(alpha0, alpha1);
  } else if (0.0f > sx || sx > nx) {
    return;
  }

  if (0.0f != ry)
  {
    alpha0 = -sy / ry;
    alpha1 = (ny - sy) / ry;
    minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
    maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
  } else if (0.0f > sy || sy > ny) {
    return;
  }

  if (0.0f != rz)
  {
    alpha0 = - sz / rz;
    alpha1 = (nz - sz) / rz;
    minAlpha = fmax(minAlpha, fmin(alpha0, alpha1));
    maxAlpha = fmin(maxAlpha, fmax(alpha0, alpha1));
  } else if (0.0f > sz || sz > nz) {
    return;
  }

  // Step 2: Cast ray if it intersects the volume

  // Trapezoidal rule (interpolating function = piecewise linear func)
  scalar_t px, py, pz;
  scalar_t proj;
  int ix, iy, iz;
  int nyz = ny * nz;

  sx = sx - 0.5f;
  sy = sy - 0.5f;
  sz = sz - 0.5f;
  volume = volume + nx * nyz * blockIdx.z;
  //dL/dV = dL/dp * dp/dV = dL/dp because projection += voxel value with factor of 1 (or the multiplicative factor at the lastline of forward pojrection)
  //projectio here is the gradient of projection

  scalar_t total_steps = (maxAlpha - minAlpha)/step;

  proj = projection[idx] / sqrt((rx * vx)*(rx * vx) + (ry * vy)*(ry * vy) + (rz * vz)*(rz * vz));
  proj = proj / total_steps;

  // if (idx == 14968){
  //   printf("minAlpha %f maxAlpha %f\n", minAlpha, maxAlpha);
  //   printf("projection index %d, orig value %f, current value %f, nx %f, ny %f, nz %f, total steps %f\n", idx, projection[idx], proj, nx, ny, nz, total_steps);
  // }

  // Entrance boundary
  // In CUDA, voxel centers are located at (xx.5, xx.5, xx.5),
  // whereas, SwVolume has voxel centers at integers.
  // For the initial interpolated value, only a half stepsize is
  //  considered in the computation.
  bool check = minAlpha < maxAlpha;
  if (minAlpha < maxAlpha) {
    px = sx + minAlpha * rx;
    py = sy + minAlpha * ry;
    pz = sz + minAlpha * rz;

    ix = lroundf(px);
    iy = lroundf(py);
    iz = lroundf(pz);
    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
      // pointer operation, add vluae of projection to the index of nyz * ix + nz * iy + iz of volume, understand that volume is a pointer

      // if (idx == 14968){
      //   printf("first backproject value %f, ix %d, iy %d, iz %d, minAlpha %f.\n", 0.5 * proj, ix, iy, iz, minAlpha);
      // }
      atomicAdd(volume + (nyz * ix + nz * iy + iz), 0.5 * proj);
    }
    minAlpha += step;
  }

  // Mid segments
  while (minAlpha < maxAlpha)
  {
    px = sx + minAlpha * rx;
    py = sy + minAlpha * ry;
    pz = sz + minAlpha * rz;

    ix = lroundf(px);
    iy = lroundf(py);
    iz = lroundf(pz);
    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
      // if (idx == 14968){
      //   printf("backproject value %f, ix %d, iy %d, iz %d, minAlpha %f.\n", proj, ix, iy, iz, minAlpha);
      // }
      atomicAdd(volume + (nyz * ix + nz * iy + iz), proj);
    }
    minAlpha += step;
  }


  if (check) {
    minAlpha -= step;
    scalar_t c = (maxAlpha - minAlpha) / step;
    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
      // if (idx == 14968){
      //   printf("last backproject1 value %f, ix %d, iy %d, iz %d, minAlpha %f.\n", 0.5f * (c - 1.0f) * proj, ix, iy, iz, minAlpha);
      // }
      atomicAdd(volume + (nyz * ix + nz * iy + iz), 0.5f * (c - 1.0f) * proj);
    }
    px = sx + maxAlpha * rx;
    py = sy + maxAlpha * ry;
    pz = sz + maxAlpha * rz;

    // The last segment of the line integral takes care of the
    // varying length.
    ix = lroundf(px);
    iy = lroundf(py);
    iz = lroundf(pz);
    if (ix >= 0 && ix < nx && iy >= 0 && iy < ny && iz >= 0 && iz < nz) {
      // if (idx == 14968){
      //   printf("last backproject2 value %f, ix %d, iy %d, iz %d, minAlpha %f.\n", 0.5f * c * proj, ix, iy, iz, minAlpha);
      // }
      atomicAdd(volume + (nyz * ix + nz * iy + iz), 0.5f * c * proj);
    }
  }
}


torch::Tensor dp_nearest_cuda_forward(
  torch::Tensor volume,
  torch::Tensor detector_shape,
  torch::Tensor ray_mat,
  torch::Tensor ray,
  torch::Tensor step_size,
  torch::Tensor voxel_size) {

  torch::Tensor projection; // projection tensor
  scalar_t *volume_ptr; // volume ptr
  scalar_t *proj_ptr; // projection ptr
  scalar_t *ray_mat_ptr; // ray direction matrix ptr
  scalar_t *ray_ptr; // ray direction ptr
  scalar_t step; // step size
  scalar_t vx, vy, vz; // voxel size
  int batch_size; // batch size
  int dh, dw; // detector shape
  int nx, ny, nz; // volume shape
  int nc; // number of channels

  // accessors
  auto d_a = detector_shape.accessor<int,1>();
  auto v_a = voxel_size.accessor<scalar_t,1>();
  auto s_a = step_size.accessor<scalar_t,1>();

  batch_size = volume.size(0); AT_ASSERT(batch_size <= MAX_BATCH_SIZE);
  nc = volume.size(1);
  nx = volume.size(2);
  ny = volume.size(3);
  nz = volume.size(4);

  dh = d_a[0];
  dw = d_a[1];
  vx = v_a[0];
  vy = v_a[1];
  vz = v_a[2];
  step = s_a[0];

  projection = torch::zeros({batch_size, nc, dh, dw}, torch::TensorOptions().
    dtype(volume.dtype()).device(volume.device()));
  volume_ptr = volume.data<scalar_t>();
  proj_ptr = projection.data<scalar_t>();
  ray_mat_ptr = ray_mat.data<scalar_t>();
  ray_ptr = ray.data<scalar_t>();

  // move data to constant memory (which supports the fastest memory access)
  cudaMemcpyToSymbol(RAY_MAT, ray_mat_ptr, sizeof(scalar_t) * batch_size * 9);
  cudaMemcpyToSymbol(RAY, ray_ptr, sizeof(scalar_t) * batch_size * 3);

  const int nblock_h = (dh + 16 - 1) / 16;
  const int nblock_w = (dw + 16 - 1) / 16;
  const dim3 block_size(16, 16, 1);
  const dim3 grid_size(nblock_h, nblock_w, batch_size * nc);

  dp_nearest_cuda_forward_kernel<<<grid_size, block_size>>>(
    volume_ptr, proj_ptr, nx, ny, nz, dh, dw, batch_size, step, vx, vy, vz
  );
  return projection;
}

torch::Tensor dp_nearest_cuda_backward(
  torch::Tensor projection,
  torch::Tensor volume_shape,
  torch::Tensor ray_mat,
  torch::Tensor ray,
  torch::Tensor step_size,
  torch::Tensor voxel_size) {

  torch::Tensor volume; // volume tensor
  scalar_t *volume_ptr; // volume ptr
  scalar_t *proj_ptr; // projection ptr
  scalar_t *ray_mat_ptr; // ray direction matrix ptr
  scalar_t *ray_ptr; // ray direction ptr
  scalar_t step; // step size
  scalar_t vx, vy, vz; // voxel size
  int batch_size; // batch size
  int dh, dw; // detector shape
  int nx, ny, nz; // volume shape
  int nc; // number of channels

  // accessors
  auto n_a = volume_shape.accessor<int,1>();
  auto v_a = voxel_size.accessor<scalar_t,1>();
  auto s_a = step_size.accessor<scalar_t,1>();

  batch_size = projection.size(0); AT_ASSERT(batch_size <= MAX_BATCH_SIZE);
  nc = projection.size(1);
  dh = projection.size(2);
  dw = projection.size(3);

  nx = n_a[0];
  ny = n_a[1];
  nz = n_a[2];
  vx = v_a[0];
  vy = v_a[1];
  vz = v_a[2];
  step = s_a[0];

  volume = torch::zeros({batch_size, nc, nx, ny, nz}, torch::TensorOptions().
    dtype(projection.dtype()).device(projection.device()));
  volume_ptr = volume.data<scalar_t>();
  proj_ptr = projection.data<scalar_t>();
  ray_mat_ptr = ray_mat.data<scalar_t>();
  ray_ptr = ray.data<scalar_t>();

  // move data to constant memory (which supports the fastest memory access)
  cudaMemcpyToSymbol(RAY_MAT, ray_mat_ptr, sizeof(scalar_t) * batch_size * 9);
  cudaMemcpyToSymbol(RAY, ray_ptr, sizeof(scalar_t) * batch_size * 3);

  const int nblock_h = (dh + 16 - 1) / 16;
  const int nblock_w = (dw + 16 - 1) / 16;
  const dim3 block_size(16, 16, 1);
  const dim3 grid_size(nblock_h, nblock_w, batch_size * nc);

  dp_nearest_cuda_backward_kernel<<<grid_size, block_size>>>(
    volume_ptr, proj_ptr, nx, ny, nz, dh, dw, batch_size, step, vx, vy, vz
  );
  return volume;
}

torch::Tensor dp_backproject_nearest_cuda_forward(
  torch::Tensor projection,
  torch::Tensor volume_shape,
  torch::Tensor ray_mat,
  torch::Tensor ray,
  torch::Tensor step_size,
  torch::Tensor voxel_size) {

  torch::Tensor volume; // volume tensor
  scalar_t *volume_ptr; // volume ptr
  scalar_t *proj_ptr; // projection ptr
  scalar_t *ray_mat_ptr; // ray direction matrix ptr
  scalar_t *ray_ptr; // ray direction ptr
  scalar_t step; // step size
  scalar_t vx, vy, vz; // voxel size
  int batch_size; // batch size
  int dh, dw; // detector shape
  int nx, ny, nz; // volume shape
  int nc; // number of channels

  // accessors
  auto n_a = volume_shape.accessor<int,1>();
  auto v_a = voxel_size.accessor<scalar_t,1>();
  auto s_a = step_size.accessor<scalar_t,1>();

  batch_size = projection.size(0); AT_ASSERT(batch_size <= MAX_BATCH_SIZE);
  nc = projection.size(1);
  dh = projection.size(2);
  dw = projection.size(3);

  nx = n_a[0];
  ny = n_a[1];
  nz = n_a[2];
  vx = v_a[0];
  vy = v_a[1];
  vz = v_a[2];
  step = s_a[0];

  volume = torch::zeros({batch_size, nc, nx, ny, nz}, torch::TensorOptions().
    dtype(projection.dtype()).device(projection.device()));
  volume_ptr = volume.data<scalar_t>();
  proj_ptr = projection.data<scalar_t>();
  ray_mat_ptr = ray_mat.data<scalar_t>();
  ray_ptr = ray.data<scalar_t>();

  // move data to constant memory (which supports the fastest memory access)
  cudaMemcpyToSymbol(RAY_MAT, ray_mat_ptr, sizeof(scalar_t) * batch_size * 9);
  cudaMemcpyToSymbol(RAY, ray_ptr, sizeof(scalar_t) * batch_size * 3);

  const int nblock_h = (dh + 16 - 1) / 16;
  const int nblock_w = (dw + 16 - 1) / 16;
  const dim3 block_size(16, 16, 1);
  const dim3 grid_size(nblock_h, nblock_w, batch_size * nc);

  dp_backproject_nearest_cuda_forward_kernel<<<grid_size, block_size>>>(
    volume_ptr, proj_ptr, nx, ny, nz, dh, dw, batch_size, step, vx, vy, vz
  );
  return volume;
}