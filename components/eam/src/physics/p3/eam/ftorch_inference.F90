module ftorch_inference

!   ! Import precision info from iso
   use, intrinsic :: iso_fortran_env, only : sp => real32, dp => real64

   ! get real kind from utils
!   use physics_utils, only: rtype,rtype8,btype

   use cam_logfile,  only: iulog

   ! Import our library for interfacing with PyTorch
   use ftorch, only : torch_model, torch_tensor, torch_kCPU, torch_delete, &
                      torch_tensor_from_array, torch_model_load, torch_model_forward

   implicit none

   ! Set working precision for reals (legacy single-precision wrappers)
   integer, parameter :: wp = sp

   public ftorch_inference_cpu, ftorch_inference_cpu_batch, &
          ftorch_inference_cpu_dp, ftorch_inference_cpu_batch_dp, &
          init_ftorch_inference

   contains

   subroutine init_ftorch_inference(model_file_path,model)
        character(len=*), intent(in) :: model_file_path
        type(torch_model), intent(inout) :: model

        write(iulog, *) 'Torch model path:',  trim(model_file_path)
        write(iulog, *) 'torch_kCPU (device):',  torch_kCPU

   ! Load ML model
        call torch_model_load(model, trim(model_file_path), torch_kCPU)
        write(iulog, *) 'Torch Model Loaded'

   end subroutine init_ftorch_inference

   subroutine ftorch_inference_cpu(model,input_data,output_data)

   ! Pass in the model
        type(torch_model), intent(in) :: model

   ! Set up Fortran data structures
        real(wp), target, intent(in)    :: input_data(:)
        real(wp), target, intent(inout) :: output_data(:)
        real(wp), pointer, contiguous :: input_p(:)
        real(wp), pointer, contiguous :: output_p(:)


   ! Set up Torch data structures
   ! a vector of input tensors (in this case we only have one), and the output tensor

        type(torch_tensor), dimension(1) :: in_tensors
        type(torch_tensor), dimension(1) :: out_tensors

   ! Test input
!        write(iulog, *) 'Inference Input:', input_data(:)

   ! Create Torch input/output tensors from the above arrays
        input_p  => input_data
        output_p => output_data
        call torch_tensor_from_array(in_tensors(1), input_p,  torch_kCPU)
        call torch_tensor_from_array(out_tensors(1), output_p, torch_kCPU)

   ! Inference
        call torch_model_forward(model, in_tensors, out_tensors)

   ! Print output for testing
   ! expected = [0.0_wp, 2.0_wp, 4.0_wp, 6.0_wp, 8.0_wp]
   !    write(iulog, *) 'Inference Output:', output_data(:)

   ! Cleanup
        call torch_delete(in_tensors)
        call torch_delete(out_tensors)

   end subroutine ftorch_inference_cpu

   subroutine ftorch_inference_cpu_batch(model,input_data,output_data)

   ! Pass in the model
        type(torch_model), intent(in) :: model

   ! Set up Fortran data structures
        real(wp), target, intent(in)    :: input_data(:,:)
        real(wp), target, intent(inout) :: output_data(:,:)
        real(wp), pointer, contiguous :: input_p(:,:)
        real(wp), pointer, contiguous :: output_p(:,:)

   ! Set up Torch data structures
        type(torch_tensor), dimension(1) :: in_tensors
        type(torch_tensor), dimension(1) :: out_tensors

   ! Create Torch input/output tensors from the above arrays
        input_p  => input_data
        output_p => output_data
        call torch_tensor_from_array(in_tensors(1), input_p,  torch_kCPU)
        call torch_tensor_from_array(out_tensors(1), output_p, torch_kCPU)

   ! Inference
        call torch_model_forward(model, in_tensors, out_tensors)

   ! Cleanup
        call torch_delete(in_tensors)
        call torch_delete(out_tensors)

   end subroutine ftorch_inference_cpu_batch

   ! ---------------------------------------------------------------------------
   ! Double-precision (float64) variants. Use these with TorchScript models that
   ! were exported with float64 weights. Mixing dtypes (e.g. passing a float32
   ! tensor into a float64 model) causes FTorch / libtorch to raise a runtime
   ! type error, so the model file dtype must match the buffer kind.
   ! ---------------------------------------------------------------------------

   subroutine ftorch_inference_cpu_dp(model,input_data,output_data)

        type(torch_model), intent(in) :: model

        real(dp), target, intent(in)    :: input_data(:)
        real(dp), target, intent(inout) :: output_data(:)
        real(dp), pointer, contiguous :: input_p(:)
        real(dp), pointer, contiguous :: output_p(:)

        type(torch_tensor), dimension(1) :: in_tensors
        type(torch_tensor), dimension(1) :: out_tensors

        input_p  => input_data
        output_p => output_data
        call torch_tensor_from_array(in_tensors(1),  input_p,  torch_kCPU)
        call torch_tensor_from_array(out_tensors(1), output_p, torch_kCPU)

        call torch_model_forward(model, in_tensors, out_tensors)

        call torch_delete(in_tensors)
        call torch_delete(out_tensors)

   end subroutine ftorch_inference_cpu_dp

   subroutine ftorch_inference_cpu_batch_dp(model,input_data,output_data)

        type(torch_model), intent(in) :: model

        real(dp), target, intent(in)    :: input_data(:,:)
        real(dp), target, intent(inout) :: output_data(:,:)
        real(dp), pointer, contiguous :: input_p(:,:)
        real(dp), pointer, contiguous :: output_p(:,:)

        type(torch_tensor), dimension(1) :: in_tensors
        type(torch_tensor), dimension(1) :: out_tensors

        input_p  => input_data
        output_p => output_data
        call torch_tensor_from_array(in_tensors(1),  input_p,  torch_kCPU)
        call torch_tensor_from_array(out_tensors(1), output_p, torch_kCPU)

        call torch_model_forward(model, in_tensors, out_tensors)

        call torch_delete(in_tensors)
        call torch_delete(out_tensors)

   end subroutine ftorch_inference_cpu_batch_dp


end module ftorch_inference
