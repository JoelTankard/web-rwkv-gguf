use crate::{
    context::Context,
    num::Float,
    tensor::{kind::ReadWrite, ops::TensorOp, TensorCpu, TensorError, TensorGpu, TensorInit, TensorInto, TensorShape},
};

pub async fn softmax_one<T: Float>(
    context: &Context,
    input: TensorCpu<T>,
) -> Result<TensorCpu<T>, TensorError> {
    if input.size() == 0 {
        return Ok(input);
    }

    let tensor: TensorGpu<_, _> = input.to(context);
    let op = TensorOp::softmax(&tensor)?;
    context.queue.submit(context.encode(&op));

    let output = tensor.back().await;
    Ok(output)
}

pub async fn softmax<T: Float>(
    context: &Context,
    input: Vec<TensorCpu<T>>,
) -> Result<Vec<TensorCpu<T>>, TensorError> {
    let mut tensors = Vec::with_capacity(input.len());
    let mut ops = Vec::with_capacity(input.len());

    for input in input.into_iter() {
        let tensor: TensorGpu<_, _> = input.to(context);
        if tensor.size() > 0 {
            ops.push(TensorOp::softmax(&tensor)?);
        }
        tensors.push(tensor);
    }
    context.queue.submit(context.encode(&TensorOp::List(ops)));

    let mut output = Vec::with_capacity(tensors.len());
    for tensor in tensors.into_iter() {
        output.push(tensor.back().await);
    }
    Ok(output)
}

/// GPU-side argmax sampling - returns token index without reading back full logits.
/// This avoids reading back 260KB of logits per token (65K vocab × 4 bytes).
/// Instead, only reads back 4 bytes (single u32 token index).
pub async fn sample_argmax_one<T: Float>(
    context: &Context,
    input: TensorCpu<T>,
) -> Result<u32, TensorError> {
    if input.size() == 0 {
        return Ok(0);
    }

    let tensor: TensorGpu<T, ReadWrite> = input.to(context);
    let shape = tensor.shape();
    
    // Create output tensor for token index
    let output: TensorGpu<u32, ReadWrite> = context.tensor_init([shape[1], shape[2], 1, 1]);
    
    let op = TensorOp::sample_argmax(&tensor, &output)?;
    context.queue.submit(context.encode(&op));

    let result = output.back().await;
    Ok(result.data()[0])
}

/// GPU-side argmax sampling for a tensor already on GPU.
/// Returns token index without reading back full logits.
pub async fn sample_argmax_gpu<T: Float>(
    context: &Context,
    tensor: &TensorGpu<T, ReadWrite>,
) -> Result<u32, TensorError> {
    if tensor.size() == 0 {
        return Ok(0);
    }

    let shape = tensor.shape();
    
    // Create output tensor for token index
    let output: TensorGpu<u32, ReadWrite> = context.tensor_init([shape[1], shape[2], 1, 1]);
    
    let op = TensorOp::sample_argmax(tensor, &output)?;
    context.queue.submit(context.encode(&op));

    let result = output.back().await;
    Ok(result.data()[0])
}
