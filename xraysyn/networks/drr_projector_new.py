import numpy as np
import torch
import drr_projector_function
from torch import nn
from torch.autograd import Function


class DRRProject(Function):
    @staticmethod
    def forward(ctx, volume, detector_shape, ray_mat, source, step_size, voxel_size, interp):
        ctx.save_for_backward(ray_mat, source, step_size, voxel_size)
        ctx.volume_shape = volume.shape[2:]
        ctx.interp = interp
        if ctx.interp == 'nearest':
            forward_function = drr_projector_function.nearest_forward
        elif ctx.interp == 'trilinear':
            forward_function = drr_projector_function.trilinear_forward
        else:
            raise ValueError(f"Invalid interpolation type: {ctx.interp}")

        projection = forward_function(volume, detector_shape, ray_mat, source, step_size, voxel_size)
        return projection

    @staticmethod
    def backward(ctx, grad_projection):
        ray_mat, source, step_size, voxel_size = ctx.saved_tensors
        volume_shape = torch.tensor(ctx.volume_shape, dtype=torch.int32)
        if ctx.interp == 'nearest':
            backward_function = drr_projector_function.nearest_backward
        elif ctx.interp == 'trilinear':
            backward_function = drr_projector_function.trilinear_backward
        else:
            raise ValueError(f"Invalid interpolation type: {ctx.interp}")

        grad_volume = backward_function(grad_projection.contiguous(), volume_shape, ray_mat, source, step_size, voxel_size)
        return grad_volume, None, None, None, None, None, None


class DRRBackProject(Function):
    @staticmethod
    def forward(ctx, projection, volume_shape, ray_mat, source, step_size, voxel_size, interp):
        ctx.save_for_backward(ray_mat, source, step_size, voxel_size)
        ctx.detector_shape = projection.shape[2:]
        ctx.interp = interp
        if ctx.interp == 'nearest':
            forward_function = drr_projector_function.nearest_backward
        elif ctx.interp == 'trilinear':
            forward_function = drr_projector_function.trilinear_backward
        else:
            raise ValueError(f"Invalid interpolation type: {ctx.interp}")

        volume = forward_function(projection, volume_shape, ray_mat, source, step_size, voxel_size)
        return volume

    @staticmethod
    def backward(ctx, grad_volume):
        ray_mat, source, step_size, voxel_size = ctx.saved_tensors
        detector_shape = torch.tensor(ctx.detector_shape, dtype=torch.int32)
        if ctx.interp == 'nearest':
            backward_function = drr_projector_function.nearest_forward
        elif ctx.interp == 'trilinear':
            backward_function = drr_projector_function.trilinear_forward
        else:
            raise ValueError(f"Invalid interpolation type: {ctx.interp}")

        grad_projection = backward_function(grad_volume.contiguous(), detector_shape, ray_mat, source, step_size, voxel_size)
        return grad_projection, None, None, None, None, None, None

class DRRSemanticBackProject(Function):
    @staticmethod
    def forward(ctx, gan_projection, gan_volume, projection, volume_shape, ray_mat, source, step_size, voxel_size, interp):
        ctx.save_for_backward(ray_mat, source, step_size, voxel_size)
        ctx.detector_shape = projection.shape[2:]
        ctx.interp = interp
        if ctx.interp == 'trilinear':
            forward_function = drr_projector_function.trilinear_semantic_backward
        elif ctx.interp == 'nearest':
            forward_function = drr_projector_function.nearest_semantic_backward
        else:
            raise ValueError(f"Invalid interpolation type: {ctx.interp}")

        volume = forward_function(gan_projection, gan_volume, projection, volume_shape, ray_mat, source, step_size, voxel_size)
        return volume

    @staticmethod
    def backward(ctx, grad_volume):
        # ray_mat, source, step_size, voxel_size = ctx.saved_tensors
        # detector_shape = torch.tensor(ctx.detector_shape, dtype=torch.int32)
        # if ctx.interp == 'nearest':
        #     backward_function = drr_projector_function.nearest_forward
        # elif ctx.interp == 'trilinear':
        #     backward_function = drr_projector_function.trilinear_forward
        # else:
        #     raise ValueError(f"Invalid interpolation type: {ctx.interp}")

        # grad_projection = backward_function(grad_volume.contiguous(), detector_shape, ray_mat, source, step_size, voxel_size)
        return None, grad_volume, None, None, None, None, None, None, None

class DRRAverageProject(Function):
    @staticmethod
    def forward(ctx, volume, detector_shape, ray_mat, source, step_size, voxel_size, interp):
        ctx.save_for_backward(ray_mat, source, step_size, voxel_size)
        ctx.volume_shape = volume.shape[2:]
        ctx.interp = interp
        if ctx.interp == 'nearest':
            forward_function = drr_projector_function.nearest_forward
        elif ctx.interp == 'trilinear':
            print("average forward")
            forward_function = drr_projector_function.trilinear_average_forward
        else:
            raise ValueError(f"Invalid interpolation type: {ctx.interp}")

        projection = forward_function(volume, detector_shape, ray_mat, source, step_size, voxel_size)
        return projection

    @staticmethod
    def backward(ctx, grad_projection):
        ray_mat, source, step_size, voxel_size = ctx.saved_tensors
        volume_shape = torch.tensor(ctx.volume_shape, dtype=torch.int32)
        if ctx.interp == 'nearest':
            backward_function = drr_projector_function.nearest_backward
        elif ctx.interp == 'trilinear':
            backward_function = drr_projector_function.trilinear_backward
        else:
            raise ValueError(f"Invalid interpolation type: {ctx.interp}")

        grad_volume = backward_function(grad_projection.contiguous(), volume_shape, ray_mat, source, step_size, voxel_size)
        return grad_volume, None, None, None, None, None, None



class DRRProjector(nn.Module):
    def __init__(
        self, mode='forward', volume_shape=(128, 128, 128), detector_shape=(128, 128),
        voxel_size=(1.0, 1.0, 1.0), pixel_size=(1.5, 1.5), step_size=0.1,
        source_to_detector_distance=1500.0, isocenter_distance=1000.0,
        source_offset=(0.0, 0.0, 0.0), detector_offset=(0.0, 0.0), interp='nearest'
    ):
        super(DRRProjector, self).__init__()

        if np.isscalar(volume_shape): volume_shape = (volume_shape,) * 3
        if np.isscalar(detector_shape): detector_shape = (detector_shape,) * 2
        if np.isscalar(voxel_size): voxel_size = (voxel_size,) * 3
        if np.isscalar(pixel_size): pixel_size = (pixel_size,) * 2
        if np.isscalar(source_offset): source_offset = (source_offset,) * 3
        if np.isscalar(detector_offset): detector_offset = (detector_offset,) * 2

        # intrinsic matrix K
        K = torch.zeros((3, 3))
        K[0, 0] = source_to_detector_distance / pixel_size[0]
        K[1, 1] = source_to_detector_distance / pixel_size[1]
        K[0, 2] = detector_shape[0] / 2.0 - detector_offset[0] / pixel_size[0]
        K[1, 2] = detector_shape[1] / 2.0 - detector_offset[1] / pixel_size[1]
        K[2, 2] = 1.0
        self.register_buffer('K_inv', K.inverse())

        # diag(s)^-1
        voxel_size_inv = torch.zeros((3, 3))
        voxel_size_inv[0, 0] = 1 / voxel_size[0]
        voxel_size_inv[1, 1] = 1 / voxel_size[1]
        voxel_size_inv[2, 2] = 1 / voxel_size[2]
        self.register_buffer('voxel_size_inv', voxel_size_inv)

        self.isocenter_distance = isocenter_distance
        volume_offset = torch.tensor(volume_shape, dtype=torch.float32) * 0.5 - \
            torch.matmul(voxel_size_inv, torch.tensor(source_offset))
        self.register_buffer("volume_offset", volume_offset)

        self.volume_shape = torch.tensor(volume_shape, dtype=torch.int32)
        self.detector_shape = torch.tensor(detector_shape, dtype=torch.int32)
        self.voxel_size = torch.tensor(voxel_size)
        self.step_size = torch.tensor([step_size])
        self.mode = mode
        self.interp = interp

    def get_device(self):
        return self.K_inv.device

    @staticmethod
    def create_rotation(theta):
        batch_size = theta.shape[0]
        device = theta.device
        dtype = theta.dtype
        theta_x, theta_y, theta_z = theta[:, 0], theta[:, 1], theta[:, 2]

        rotate_x = torch.zeros(batch_size, 3, 3, dtype=dtype, device=device)
        rotate_x[:, 0, 0] = 1.0
        rotate_x[:, 1, 1] = torch.cos(theta_x)
        rotate_x[:, 1, 2] = -torch.sin(theta_x)
        rotate_x[:, 2, 1] = torch.sin(theta_x)
        rotate_x[:, 2, 2] = torch.cos(theta_x)

        rotate_y = torch.zeros(batch_size, 3, 3, dtype=dtype, device=device)
        rotate_y[:, 0, 0] = torch.cos(theta_y)
        rotate_y[:, 0, 2] = torch.sin(theta_y)
        rotate_y[:, 1, 1] = 1.0
        rotate_y[:, 2, 0] = -torch.sin(theta_y)
        rotate_y[:, 2, 2] = torch.cos(theta_y)

        rotate_z = torch.zeros(batch_size, 3, 3, dtype=dtype, device=device)
        rotate_z[:, 0, 0] = torch.cos(theta_z)
        rotate_z[:, 0, 1] = -torch.sin(theta_z)
        rotate_z[:, 1, 0] = torch.sin(theta_z)
        rotate_z[:, 1, 1] = torch.cos(theta_z)
        rotate_z[:, 2, 2] = 1.0

        return torch.matmul(torch.matmul(rotate_x, rotate_y), rotate_z)

    def forward(self, input_data, transform_param=None):
        assert input_data.is_cuda, "Only GPU tensors are supported!"

        dtype = input_data.dtype
        device = input_data.device
        voxel_size_inv = self.voxel_size_inv.to(dtype)
        K_inv = self.K_inv.to(dtype)
        volume_offset = self.volume_offset.to(dtype)
        step_size = self.step_size.to(dtype)
        voxel_size = self.voxel_size.to(dtype)
        if transform_param is None:
            transform_param = torch.zeros(
                input_data.shape[0], 6, dtype=dtype, device=device)

        if transform_param.dim() == 2:
            R = DRRProjector.create_rotation(transform_param[:, :3])
            t = -transform_param[:, 3:][..., np.newaxis]
        else:
            R = transform_param[:, :3, :3]
            t = -transform_param[:, :3, 3][..., np.newaxis]
        t[:, 2] = self.isocenter_distance + t[:, 2]

        ray_mat = torch.matmul(voxel_size_inv, torch.matmul(R.transpose(1, 2), K_inv))
        source = volume_offset - \
            torch.matmul(voxel_size_inv, torch.matmul(R.transpose(1, 2), t)).squeeze(-1)

        if self.mode == "forward":
            assert tuple(input_data.shape[2:]) == tuple(self.volume_shape), \
                "Input data shape does not match volume shape."

            projection = DRRProject.apply(input_data, self.detector_shape, ray_mat,
                source, step_size, voxel_size, self.interp)
            return projection / 10.0
        elif self.mode == "backward":
            assert tuple(input_data.shape[2:]) == tuple(self.detector_shape), \
                "Input data shape does not match detector shape."

            volume = DRRBackProject.apply(input_data * 10.0, self.volume_shape, ray_mat,
                source, step_size, voxel_size, self.interp)
            return volume
        else:
            raise ValueError(f"Invalid projection mode: {self.mode}")



class DRRAverageProjector(nn.Module):
    def __init__(
        self, volume_shape=(128, 128, 128), detector_shape=(128, 128),
        voxel_size=(1.0, 1.0, 1.0), pixel_size=(1.5, 1.5), step_size=0.1,
        source_to_detector_distance=1500.0, isocenter_distance=1000.0,
        source_offset=(0.0, 0.0, 0.0), detector_offset=(0.0, 0.0), interp='nearest'
    ):
        super(DRRAverageProjector, self).__init__()

        if np.isscalar(volume_shape): volume_shape = (volume_shape,) * 3
        if np.isscalar(detector_shape): detector_shape = (detector_shape,) * 2
        if np.isscalar(voxel_size): voxel_size = (voxel_size,) * 3
        if np.isscalar(pixel_size): pixel_size = (pixel_size,) * 2
        if np.isscalar(source_offset): source_offset = (source_offset,) * 3
        if np.isscalar(detector_offset): detector_offset = (detector_offset,) * 2
        mode = 'forward'
        # intrinsic matrix K
        K = torch.zeros((3, 3))
        K[0, 0] = source_to_detector_distance / pixel_size[0]
        K[1, 1] = source_to_detector_distance / pixel_size[1]
        K[0, 2] = detector_shape[0] / 2.0 - detector_offset[0] / pixel_size[0]
        K[1, 2] = detector_shape[1] / 2.0 - detector_offset[1] / pixel_size[1]
        K[2, 2] = 1.0
        self.register_buffer('K_inv', K.inverse())

        # diag(s)^-1
        voxel_size_inv = torch.zeros((3, 3))
        voxel_size_inv[0, 0] = 1 / voxel_size[0]
        voxel_size_inv[1, 1] = 1 / voxel_size[1]
        voxel_size_inv[2, 2] = 1 / voxel_size[2]
        self.register_buffer('voxel_size_inv', voxel_size_inv)

        self.isocenter_distance = isocenter_distance
        volume_offset = torch.tensor(volume_shape, dtype=torch.float32) * 0.5 - \
            torch.matmul(voxel_size_inv, torch.tensor(source_offset))
        self.register_buffer("volume_offset", volume_offset)

        self.volume_shape = torch.tensor(volume_shape, dtype=torch.int32)
        self.detector_shape = torch.tensor(detector_shape, dtype=torch.int32)
        self.voxel_size = torch.tensor(voxel_size)
        self.step_size = torch.tensor([step_size])
        self.mode = mode
        self.interp = interp

    def get_device(self):
        return self.K_inv.device

    @staticmethod
    def create_rotation(theta):
        batch_size = theta.shape[0]
        device = theta.device
        dtype = theta.dtype
        theta_x, theta_y, theta_z = theta[:, 0], theta[:, 1], theta[:, 2]

        rotate_x = torch.zeros(batch_size, 3, 3, dtype=dtype, device=device)
        rotate_x[:, 0, 0] = 1.0
        rotate_x[:, 1, 1] = torch.cos(theta_x)
        rotate_x[:, 1, 2] = -torch.sin(theta_x)
        rotate_x[:, 2, 1] = torch.sin(theta_x)
        rotate_x[:, 2, 2] = torch.cos(theta_x)

        rotate_y = torch.zeros(batch_size, 3, 3, dtype=dtype, device=device)
        rotate_y[:, 0, 0] = torch.cos(theta_y)
        rotate_y[:, 0, 2] = torch.sin(theta_y)
        rotate_y[:, 1, 1] = 1.0
        rotate_y[:, 2, 0] = -torch.sin(theta_y)
        rotate_y[:, 2, 2] = torch.cos(theta_y)

        rotate_z = torch.zeros(batch_size, 3, 3, dtype=dtype, device=device)
        rotate_z[:, 0, 0] = torch.cos(theta_z)
        rotate_z[:, 0, 1] = -torch.sin(theta_z)
        rotate_z[:, 1, 0] = torch.sin(theta_z)
        rotate_z[:, 1, 1] = torch.cos(theta_z)
        rotate_z[:, 2, 2] = 1.0

        return torch.matmul(torch.matmul(rotate_x, rotate_y), rotate_z)

    def forward(self, input_data, transform_param=None):
        assert input_data.is_cuda, "Only GPU tensors are supported!"

        dtype = input_data.dtype
        device = input_data.device
        voxel_size_inv = self.voxel_size_inv.to(dtype)
        K_inv = self.K_inv.to(dtype)
        volume_offset = self.volume_offset.to(dtype)
        step_size = self.step_size.to(dtype)
        voxel_size = self.voxel_size.to(dtype)
        if transform_param is None:
            transform_param = torch.zeros(
                input_data.shape[0], 6, dtype=dtype, device=device)

        if transform_param.dim() == 2:
            R = DRRProjector.create_rotation(transform_param[:, :3])
            t = -transform_param[:, 3:][..., np.newaxis]
        else:
            R = transform_param[:, :3, :3]
            t = -transform_param[:, :3, 3][..., np.newaxis]
        t[:, 2] = self.isocenter_distance + t[:, 2]

        ray_mat = torch.matmul(voxel_size_inv, torch.matmul(R.transpose(1, 2), K_inv))
        source = volume_offset - \
            torch.matmul(voxel_size_inv, torch.matmul(R.transpose(1, 2), t)).squeeze(-1)


        assert tuple(input_data.shape[2:]) == tuple(self.volume_shape), \
            "Input data shape does not match volume shape."

        projection = DRRAverageProject.apply(input_data, self.detector_shape, ray_mat,
            source, step_size, voxel_size, self.interp)
        return projection / 10.0



class DRRSemanticBackProjector(nn.Module):
    def __init__(
        self, volume_shape=(128, 128, 128), detector_shape=(128, 128),
        voxel_size=(1.0, 1.0, 1.0), pixel_size=(1.5, 1.5), step_size=0.1,
        source_to_detector_distance=1500.0, isocenter_distance=1000.0,
        source_offset=(0.0, 0.0, 0.0), detector_offset=(0.0, 0.0), interp='nearest'
    ):
        super(DRRSemanticBackProjector, self).__init__()

        if np.isscalar(volume_shape): volume_shape = (volume_shape,) * 3
        if np.isscalar(detector_shape): detector_shape = (detector_shape,) * 2
        if np.isscalar(voxel_size): voxel_size = (voxel_size,) * 3
        if np.isscalar(pixel_size): pixel_size = (pixel_size,) * 2
        if np.isscalar(source_offset): source_offset = (source_offset,) * 3
        if np.isscalar(detector_offset): detector_offset = (detector_offset,) * 2
        mode = 'backward'
        # intrinsic matrix K
        K = torch.zeros((3, 3))
        K[0, 0] = source_to_detector_distance / pixel_size[0]
        K[1, 1] = source_to_detector_distance / pixel_size[1]
        K[0, 2] = detector_shape[0] / 2.0 - detector_offset[0] / pixel_size[0]
        K[1, 2] = detector_shape[1] / 2.0 - detector_offset[1] / pixel_size[1]
        K[2, 2] = 1.0
        self.register_buffer('K_inv', K.inverse())

        # diag(s)^-1
        voxel_size_inv = torch.zeros((3, 3))
        voxel_size_inv[0, 0] = 1 / voxel_size[0]
        voxel_size_inv[1, 1] = 1 / voxel_size[1]
        voxel_size_inv[2, 2] = 1 / voxel_size[2]
        self.register_buffer('voxel_size_inv', voxel_size_inv)

        self.isocenter_distance = isocenter_distance
        volume_offset = torch.tensor(volume_shape, dtype=torch.float32) * 0.5 - \
            torch.matmul(voxel_size_inv, torch.tensor(source_offset))
        self.register_buffer("volume_offset", volume_offset)

        self.volume_shape = torch.tensor(volume_shape, dtype=torch.int32)
        self.detector_shape = torch.tensor(detector_shape, dtype=torch.int32)
        self.voxel_size = torch.tensor(voxel_size)
        self.step_size = torch.tensor([step_size])
        self.mode = mode
        self.interp = interp

    def get_device(self):
        return self.K_inv.device

    @staticmethod
    def create_rotation(theta):
        batch_size = theta.shape[0]
        device = theta.device
        dtype = theta.dtype
        theta_x, theta_y, theta_z = theta[:, 0], theta[:, 1], theta[:, 2]

        rotate_x = torch.zeros(batch_size, 3, 3, dtype=dtype, device=device)
        rotate_x[:, 0, 0] = 1.0
        rotate_x[:, 1, 1] = torch.cos(theta_x)
        rotate_x[:, 1, 2] = -torch.sin(theta_x)
        rotate_x[:, 2, 1] = torch.sin(theta_x)
        rotate_x[:, 2, 2] = torch.cos(theta_x)

        rotate_y = torch.zeros(batch_size, 3, 3, dtype=dtype, device=device)
        rotate_y[:, 0, 0] = torch.cos(theta_y)
        rotate_y[:, 0, 2] = torch.sin(theta_y)
        rotate_y[:, 1, 1] = 1.0
        rotate_y[:, 2, 0] = -torch.sin(theta_y)
        rotate_y[:, 2, 2] = torch.cos(theta_y)

        rotate_z = torch.zeros(batch_size, 3, 3, dtype=dtype, device=device)
        rotate_z[:, 0, 0] = torch.cos(theta_z)
        rotate_z[:, 0, 1] = -torch.sin(theta_z)
        rotate_z[:, 1, 0] = torch.sin(theta_z)
        rotate_z[:, 1, 1] = torch.cos(theta_z)
        rotate_z[:, 2, 2] = 1.0

        return torch.matmul(torch.matmul(rotate_x, rotate_y), rotate_z)

    def forward(self, gan_projection, gan_volume, input_data, transform_param=None):
        assert input_data.is_cuda, "Only GPU tensors are supported!"

        dtype = input_data.dtype
        device = input_data.device
        voxel_size_inv = self.voxel_size_inv.to(dtype)
        K_inv = self.K_inv.to(dtype)
        volume_offset = self.volume_offset.to(dtype)
        step_size = self.step_size.to(dtype)
        voxel_size = self.voxel_size.to(dtype)
        if transform_param is None:
            transform_param = torch.zeros(
                input_data.shape[0], 6, dtype=dtype, device=device)

        if transform_param.dim() == 2:
            R = DRRProjector.create_rotation(transform_param[:, :3])
            t = -transform_param[:, 3:][..., np.newaxis]
        else:
            R = transform_param[:, :3, :3]
            t = -transform_param[:, :3, 3][..., np.newaxis]
        t[:, 2] = self.isocenter_distance + t[:, 2]

        ray_mat = torch.matmul(voxel_size_inv, torch.matmul(R.transpose(1, 2), K_inv))
        source = volume_offset - \
            torch.matmul(voxel_size_inv, torch.matmul(R.transpose(1, 2), t)).squeeze(-1)

        assert tuple(input_data.shape[2:]) == tuple(self.detector_shape), \
            "Input data shape does not match detector shape."

        volume = DRRSemanticBackProject.apply(gan_projection * 10.0, gan_volume, input_data * 10.0, self.volume_shape, ray_mat,
            source, step_size, voxel_size, self.interp)
        return volume