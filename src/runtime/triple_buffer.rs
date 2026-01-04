//! Triple-buffered runtime for overlapping CPU and GPU work.
//!
//! Overlaps job building (CPU) with GPU execution by speculatively
//! pre-building the next job while the current one executes.
//!
//! This hides the ~7ms dispatch time by doing it in parallel with GPU execution.

#[cfg(not(target_arch = "wasm32"))]
use futures::future::BoxFuture;

use super::{infer, Dispatcher, Job, JobInfo, JobInput, RuntimeError};

#[cfg(all(not(target_arch = "wasm32"), feature = "tokio"))]
#[derive(Debug)]
struct Submission<I: infer::Infer> {
    input: I::Input,
    sender: flume::Sender<Result<(I::Input, I::Output), RuntimeError>>,
}

#[cfg(all(not(target_arch = "wasm32"), feature = "tokio"))]
#[derive(Debug, Clone)]
pub struct TripleBufferRuntime<I: infer::Infer>(flume::Sender<Submission<I>>);

#[cfg(all(not(target_arch = "wasm32"), feature = "tokio"))]
impl<I, T, F> TripleBufferRuntime<I>
where
    I: infer::Infer,
    T: JobInfo,
    F: Iterator<Item = T> + Send + 'static,
    for<'a> &'a I::Input: IntoIterator<Item = T, IntoIter = F>,
{
    pub async fn new<M, J>(bundle: M) -> Self
    where
        M: Dispatcher<J, Info = T> + Send + Sync + 'static,
        J: Job<Input = I::Input, Output = I::Output> + Send + 'static,
    {
        let (sender, receiver) = flume::bounded(1);
        let handle = tokio::spawn(Self::run(bundle.into(), receiver));
        tokio::spawn(async move {
            if let Err(err) = handle.await {
                log::error!("{err}");
            }
        });
        Self(sender)
    }

    async fn run<M, J>(model: std::sync::Arc<M>, receiver: flume::Receiver<Submission<I>>)
    where
        M: Dispatcher<J, Info = T> + Send + Sync + 'static,
        J: Job<Input = I::Input, Output = I::Output> + Send + 'static,
    {
        // Cache for pre-built job
        let mut cached_job: Option<(T, J)> = None;

        'main: while let Ok(Submission { input, sender }) = receiver.recv_async().await {
            let Some(info) = (&input).into_iter().next() else {
                let _ = sender.send(Err(RuntimeError::InputExhausted));
                continue 'main;
            };

            let chunk = input.chunk();

            // Step 1: Get job from cache or build new one
            let mut job = if let Some((cached_info, job)) = cached_job.take() {
                if info.check(&cached_info) {
                    job
                } else {
                    // Cache miss, build synchronously
                    let model_clone = model.clone();
                    let info_clone = info.clone();
                    match tokio::task::spawn_blocking(move || model_clone.dispatch(info_clone)).await {
                        Ok(Ok(j)) => j,
                        Ok(Err(e)) => {
                            let _ = sender.send(Err(e));
                            continue 'main;
                        }
                        Err(e) => {
                            let _ = sender.send(Err(RuntimeError::from(e)));
                            continue 'main;
                        }
                    }
                }
            } else {
                // No cached job, build synchronously
                let model_clone = model.clone();
                let info_clone = info.clone();
                match tokio::task::spawn_blocking(move || model_clone.dispatch(info_clone)).await {
                    Ok(Ok(j)) => j,
                    Ok(Err(e)) => {
                        let _ = sender.send(Err(e));
                        continue 'main;
                    }
                    Err(e) => {
                        let _ = sender.send(Err(RuntimeError::from(e)));
                        continue 'main;
                    }
                }
            };

            // Step 2: Load embeddings and submit to GPU
            if let Err(error) = job.load(&chunk) {
                let _ = sender.send(Err(error));
                continue 'main;
            }
            job.submit();

            // Step 3: While GPU is executing, build next job speculatively
            let model_clone = model.clone();
            let info_clone = info.clone();
            let build_handle = tokio::task::spawn_blocking(move || {
                model_clone.dispatch(info_clone.clone()).map(|j| (info_clone, j))
            });

            // Step 4: Wait for GPU to finish and read back results
            let output = job.back().await;

            // Step 5: Store the pre-built job for next iteration
            if let Ok(Ok(built)) = build_handle.await {
                cached_job = Some(built);
            }

            // Step 6: Send result to caller
            let mut input = input;
            input.step();
            let _ = sender.send(output.map(|output| (input, output)));
        }
    }

    pub async fn infer(&self, input: I::Input) -> Result<(I::Input, I::Output), RuntimeError> {
        let (sender, receiver) = flume::bounded(1);
        let submission = Submission { input, sender };
        let _ = self.0.send_async(submission).await;
        receiver.recv_async().await?
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "tokio"))]
impl<I, T, F> super::Runtime<I> for TripleBufferRuntime<I>
where
    I: infer::Infer,
    T: JobInfo,
    F: Iterator<Item = T> + Send + 'static,
    for<'a> &'a I::Input: IntoIterator<Item = T, IntoIter = F>,
{
    fn infer(&self, input: I::Input) -> BoxFuture<'_, Result<(I::Input, I::Output), RuntimeError>> {
        Box::pin(self.infer(input))
    }
}
