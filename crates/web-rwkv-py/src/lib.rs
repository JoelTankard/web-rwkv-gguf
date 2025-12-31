use std::{
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
    sync::Arc,
};

use anyhow::{bail, Result};
use derivative::Derivative;
use half::f16;
use memmap2::Mmap;
use numpy::{IntoPyArray, PyArray1};
use pyo3::{exceptions::PyValueError, prelude::*};
use safetensors::SafeTensors;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        gguf::GgufReader,
        infer::{Rnn, RnnInput, RnnInputBatch, RnnOption},
        loader::Loader,
        model::{
            Bundle, ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant,
            State as ModelState,
        },
        v4, v5, v6, v7, TokioRuntime,
    },
    tensor::{
        kind::ReadWrite, DeepClone, TensorCpu, TensorGpu, TensorInit, TensorInto, TensorShape,
    },
    wgpu,
};

pub mod info;

fn err(err: impl ToString) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyclass]
#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct Model {
    tokio: Arc<tokio::runtime::Runtime>,
    info: ModelInfo,
    context: Context,
    runtime: TokioRuntime<Rnn>,
    #[derivative(Debug = "ignore")]
    state: Arc<dyn ModelState + Send + Sync>,
}

#[pyclass]
#[derive(Debug, Clone)]
pub enum State {
    Cpu { state: StateCpu },
    Gpu { state: StateGpu },
}

#[pymethods]
impl State {
    pub fn deep_clone(&self) -> Self {
        match self.clone() {
            State::Cpu { state } => State::Cpu { state },
            State::Gpu { state } => {
                let state = StateGpu(state.0.deep_clone());
                State::Gpu { state }
            }
        }
    }

    pub fn device(&self) -> StateDevice {
        match self {
            State::Cpu { .. } => StateDevice::Cpu,
            State::Gpu { .. } => StateDevice::Gpu,
        }
    }

    pub fn to(&self, device: StateDevice) -> Self {
        match (self.clone(), device) {
            (Self::Cpu { state }, StateDevice::Gpu) => {
                let StateCpu(tensor, context) = state;
                let state = StateGpu(tensor.to(&context));
                Self::Gpu { state }
            }
            (Self::Gpu { state }, StateDevice::Cpu) => {
                let context = state.0.context.clone();
                let tensor = state.0.back_in_place();
                let state = StateCpu(tensor, context);
                Self::Cpu { state }
            }
            (state, _) => state,
        }
    }

    pub fn to_list(&self) -> PyResult<Vec<f32>> {
        match self {
            State::Cpu { state } => Ok(state.0.to_vec()),
            State::Gpu { state } => {
                let tensor = state.0.back_in_place();
                Ok(tensor.to_vec())
            }
        }
    }

    /// Returns state data as a numpy array (faster than to_list)
    pub fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let data = match self {
            State::Cpu { state } => state.0.to_vec(),
            State::Gpu { state } => {
                let tensor = state.0.back_in_place();
                tensor.to_vec()
            }
        };
        Ok(data.into_pyarray(py))
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct StateCpu(TensorCpu<f32>, Context);

#[pyclass]
#[derive(Debug, Clone)]
pub struct StateGpu(TensorGpu<f32, ReadWrite>);

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateDevice {
    Cpu,
    Gpu,
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyRnnOption {
    Last,
    Full,
    EmbedLast,
    EmbedFull,
}

impl From<PyRnnOption> for RnnOption {
    fn from(opt: PyRnnOption) -> Self {
        match opt {
            PyRnnOption::Last => RnnOption::Last,
            PyRnnOption::Full => RnnOption::Full,
            PyRnnOption::EmbedLast => RnnOption::EmbedLast,
            PyRnnOption::EmbedFull => RnnOption::EmbedFull,
        }
    }
}

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .auto_limits(info)
        .build()
        .await?;
    Ok(context)
}

async fn load_runtime(
    path: PathBuf,
    quant: usize,
    quant_nf4: usize,
    quant_sf4: usize,
) -> Result<(
    Context,
    ModelInfo,
    TokioRuntime<Rnn>,
    Arc<dyn ModelState + Send + Sync>,
)> {
    let file = File::open(&path)?;
    let data = unsafe { Mmap::map(&file)? };

    let is_gguf = path.extension().map_or(false, |ext| ext == "gguf");

    let quant_map = (0..quant)
        .map(|layer| (layer, Quant::Int8))
        .chain((0..quant_nf4).map(|layer| (layer, Quant::NF4)))
        .chain((0..quant_sf4).map(|layer| (layer, Quant::SF4)))
        .collect();

    if is_gguf {
        let model =
            GgufReader::new(&data).map_err(|e| anyhow::anyhow!("Failed to parse GGUF: {:?}", e))?;
        let info = Loader::info(&model)?;
        let context = create_context(&info).await?;
        let builder = ModelBuilder::new(&context, model).quant(quant_map);

        match info.version {
            ModelVersion::V4 => {
                let model = builder.build_v4().await?;
                let bundle = v4::Bundle::<f16>::new(model, 1);
                let state = Arc::new(bundle.state());
                let runtime = TokioRuntime::new(bundle).await;
                Ok((context, info, runtime, state))
            }
            ModelVersion::V5 => {
                let model = builder.build_v5().await?;
                let bundle = v5::Bundle::<f16>::new(model, 1);
                let state = Arc::new(bundle.state());
                let runtime = TokioRuntime::new(bundle).await;
                Ok((context, info, runtime, state))
            }
            ModelVersion::V6 => {
                let model = builder.build_v6().await?;
                let bundle = v6::Bundle::<f16>::new(model, 1);
                let state = Arc::new(bundle.state());
                let runtime = TokioRuntime::new(bundle).await;
                Ok((context, info, runtime, state))
            }
            ModelVersion::V7 => {
                let model = builder.build_v7().await?;
                let bundle = v7::Bundle::<f16>::new(model, 1);
                let state = Arc::new(bundle.state());
                let runtime = TokioRuntime::new(bundle).await;
                Ok((context, info, runtime, state))
            }
        }
    } else {
        let model = SafeTensors::deserialize(&data)?;
        let info = Loader::info(&model)?;
        let context = create_context(&info).await?;
        let builder = ModelBuilder::new(&context, model).quant(quant_map);

        match info.version {
            ModelVersion::V4 => {
                let model = builder.build_v4().await?;
                let bundle = v4::Bundle::<f16>::new(model, 1);
                let state = Arc::new(bundle.state());
                let runtime = TokioRuntime::new(bundle).await;
                Ok((context, info, runtime, state))
            }
            ModelVersion::V5 => {
                let model = builder.build_v5().await?;
                let bundle = v5::Bundle::<f16>::new(model, 1);
                let state = Arc::new(bundle.state());
                let runtime = TokioRuntime::new(bundle).await;
                Ok((context, info, runtime, state))
            }
            ModelVersion::V6 => {
                let model = builder.build_v6().await?;
                let bundle = v6::Bundle::<f16>::new(model, 1);
                let state = Arc::new(bundle.state());
                let runtime = TokioRuntime::new(bundle).await;
                Ok((context, info, runtime, state))
            }
            ModelVersion::V7 => {
                let model = builder.build_v7().await?;
                let bundle = v7::Bundle::<f16>::new(model, 1);
                let state = Arc::new(bundle.state());
                let runtime = TokioRuntime::new(bundle).await;
                Ok((context, info, runtime, state))
            }
        }
    }
}

#[pymethods]
impl Model {
    #[new]
    #[pyo3(signature = (path, quant=0, quant_nf4=0, quant_sf4=0))]
    pub fn new(path: PathBuf, quant: usize, quant_nf4: usize, quant_sf4: usize) -> PyResult<Self> {
        let tokio = Arc::new(tokio::runtime::Runtime::new()?);
        let (context, info, runtime, state) = tokio
            .block_on(load_runtime(path, quant, quant_nf4, quant_sf4))
            .map_err(err)?;
        Ok(Self {
            tokio,
            context,
            info,
            runtime,
            state,
        })
    }

    pub fn info(&self) -> info::ModelInfo {
        self.info.clone().into()
    }

    pub fn init_state(&self) -> State {
        let state = StateCpu(self.state.init(), self.context.clone());
        State::Cpu { state }
    }

    pub fn clear_state(&self) {
        let _ = self.load_state(&self.init_state());
    }

    pub fn load_state(&self, state: &State) -> PyResult<()> {
        match state.clone() {
            State::Cpu { state } => self.state.load(state.0, 0),
            State::Gpu { state } => self.state.write(state.0, 0),
        }
        .map_err(err)
    }

    #[pyo3(signature = (device=StateDevice::Cpu))]
    pub fn back_state(&self, device: StateDevice) -> PyResult<State> {
        match device {
            StateDevice::Cpu => {
                let tensor = self.tokio.block_on(self.state.back(0)).map_err(err)?;
                let state = StateCpu(tensor, self.context.clone());
                Ok(State::Cpu { state })
            }
            StateDevice::Gpu => {
                let tensor = self.state.read(0).map_err(err)?;
                let state = StateGpu(tensor);
                Ok(State::Gpu { state })
            }
        }
    }

    /// Get state as numpy array directly (faster than back_state().to_numpy())
    #[pyo3(signature = (device=StateDevice::Cpu))]
    pub fn back_state_numpy<'py>(
        &self,
        py: Python<'py>,
        device: StateDevice,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let state = self.back_state(device)?;
        state.to_numpy(py)
    }

    #[pyo3(signature = (tokens, token_chunk_size=128, option=None))]
    pub fn run(
        &self,
        tokens: Vec<u16>,
        token_chunk_size: usize,
        option: Option<PyRnnOption>,
    ) -> PyResult<Vec<f32>> {
        let model = self.clone();
        let option = option.map(RnnOption::from).unwrap_or(RnnOption::Last);
        let output = self
            .tokio
            .block_on(model.run_internal(tokens, option, token_chunk_size))
            .map_err(err)?
            .to_vec();
        Ok(output)
    }

    #[pyo3(signature = (tokens, token_chunk_size=128))]
    pub fn run_full(&self, tokens: Vec<u16>, token_chunk_size: usize) -> PyResult<Vec<f32>> {
        let model = self.clone();
        let option = RnnOption::Full;
        let output = self
            .tokio
            .block_on(model.run_internal(tokens, option, token_chunk_size))
            .map_err(err)?
            .to_vec();
        Ok(output)
    }

    /// Extract embeddings without running head projection (faster for embedding extraction)
    #[pyo3(signature = (tokens, token_chunk_size=128, last_only=true))]
    pub fn embed<'py>(
        &self,
        py: Python<'py>,
        tokens: Vec<u16>,
        token_chunk_size: usize,
        last_only: bool,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let model = self.clone();
        let option = if last_only {
            RnnOption::EmbedLast
        } else {
            RnnOption::EmbedFull
        };
        let output = self
            .tokio
            .block_on(model.run_internal(tokens, option, token_chunk_size))
            .map_err(err)?;
        Ok(output.to_vec().into_pyarray(py))
    }

    /// Extract embeddings for multiple sequences in a single batched GPU pass.
    /// This is significantly faster than calling embed() multiple times.
    ///
    /// Args:
    ///     sequences: List of token sequences to embed
    ///     token_chunk_size: Chunk size for processing (default 512 for better throughput)
    ///
    /// Returns:
    ///     List of embedding vectors, one per input sequence
    #[pyo3(signature = (sequences, token_chunk_size=512))]
    pub fn embed_batch<'py>(
        &self,
        py: Python<'py>,
        sequences: Vec<Vec<u16>>,
        token_chunk_size: usize,
    ) -> PyResult<Vec<Bound<'py, PyArray1<f32>>>> {
        let model = self.clone();
        let outputs = self
            .tokio
            .block_on(model.run_batch_internal(sequences, RnnOption::EmbedLast, token_chunk_size))
            .map_err(err)?;

        Ok(outputs
            .into_iter()
            .map(|t| t.to_vec().into_pyarray(py))
            .collect())
    }
}

impl Model {
    async fn run_internal(
        &self,
        tokens: Vec<u16>,
        option: RnnOption,
        token_chunk_size: usize,
    ) -> Result<TensorCpu<f32>> {
        if tokens.is_empty() {
            bail!("input tokens cannot be empty")
        }

        let tokens: Vec<u32> = tokens.into_iter().map(|t| t as u32).collect();
        let mut inference = Some(RnnInput::new(
            vec![RnnInputBatch::new(tokens, option)],
            token_chunk_size,
        ));
        let mut data = vec![];
        let mut num_token = 0;
        loop {
            let input = inference.take().unwrap();
            if input.batches[0].tokens.is_empty() {
                break;
            }

            let (input, output) = match self.runtime.infer(input).await {
                Ok(output) => output,
                Err(err) => {
                    bail!(err);
                }
            };

            num_token += output[0].0.shape()[1];
            let mut output = output[0].0.clone().to_vec();
            data.append(&mut output);
            inference.replace(input);
        }

        let num_vocab = self.info.num_vocab;
        let tensor = TensorCpu::from_data([num_vocab, num_token, 1, 1], data)?;
        Ok(tensor)
    }

    async fn run_batch_internal(
        &self,
        sequences: Vec<Vec<u16>>,
        option: RnnOption,
        token_chunk_size: usize,
    ) -> Result<Vec<TensorCpu<f32>>> {
        if sequences.is_empty() {
            bail!("sequences cannot be empty")
        }
        if sequences.iter().any(|s| s.is_empty()) {
            bail!("each sequence must have at least one token")
        }

        let num_batch = sequences.len();
        let batches: Vec<RnnInputBatch> = sequences
            .into_iter()
            .map(|tokens| {
                let tokens: Vec<u32> = tokens.into_iter().map(|t| t as u32).collect();
                RnnInputBatch::new(tokens, option)
            })
            .collect();

        let mut inference = Some(RnnInput::new(batches, token_chunk_size));
        let mut batch_data: Vec<Vec<f32>> = vec![vec![]; num_batch];
        let mut batch_tokens: Vec<usize> = vec![0; num_batch];

        loop {
            let input = inference.take().unwrap();
            let all_empty = input.batches.iter().all(|b| b.tokens.is_empty());
            if all_empty {
                break;
            }

            let (input, output) = match self.runtime.infer(input).await {
                Ok(output) => output,
                Err(err) => {
                    bail!(err);
                }
            };

            for (i, batch_output) in output.0.iter().enumerate() {
                let tokens_in_batch = batch_output.0.shape()[1];
                if tokens_in_batch > 0 {
                    batch_tokens[i] += tokens_in_batch;
                    let mut output_data = batch_output.0.clone().to_vec();
                    batch_data[i].append(&mut output_data);
                }
            }
            inference.replace(input);
        }

        let num_emb = self.info.num_emb;
        let results: Result<Vec<_>> = batch_data
            .into_iter()
            .zip(batch_tokens.iter())
            .map(|(data, &num_token)| {
                TensorCpu::from_data([num_emb, num_token, 1, 1], data)
                    .map_err(|e| anyhow::anyhow!("{}", e))
            })
            .collect();

        results
    }
}

fn load_tokenizer(path: PathBuf) -> Result<web_rwkv::tokenizer::Tokenizer> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents)?;
    Ok(web_rwkv::tokenizer::Tokenizer::new(&contents)?)
}

#[pyclass]
pub struct Tokenizer(web_rwkv::tokenizer::Tokenizer);

#[pymethods]
impl Tokenizer {
    #[new]
    pub fn new(path: PathBuf) -> PyResult<Self> {
        Ok(Self(load_tokenizer(path).map_err(err)?))
    }

    pub fn encode(&self, text: &str) -> PyResult<Vec<u16>> {
        let tokens: Vec<u32> = self.0.encode(text.as_bytes()).map_err(err)?;
        Ok(tokens.into_iter().map(|t| t as u16).collect())
    }

    pub fn decode(&self, tokens: Vec<u16>) -> PyResult<Vec<u8>> {
        let tokens: Vec<u32> = tokens.into_iter().map(|t| t as u32).collect();
        self.0.decode(&tokens).map_err(err)
    }
}

#[pymodule]
fn web_rwkv_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Model>()?;
    module.add_class::<State>()?;
    module.add_class::<StateDevice>()?;
    module.add_class::<PyRnnOption>()?;
    module.add_class::<Tokenizer>()?;
    module.add_class::<info::ModelInfo>()?;
    module.add_class::<info::ModelVersion>()?;

    Ok(())
}
