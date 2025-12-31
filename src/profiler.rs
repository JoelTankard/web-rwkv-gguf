use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, QuerySet, Queue};

const MAX_QUERIES: u32 = 256;

#[derive(Debug, Clone)]
pub struct TimingResult {
    pub name: String,
    pub duration_ns: u64,
}

#[derive(Debug)]
struct ProfilerInner {
    query_set: QuerySet,
    resolve_buffer: Buffer,
    read_buffer: Buffer,
    timestamp_period: f32,
    query_names: Vec<String>,
    query_count: u32,
    enabled: bool,
}

#[derive(Debug, Clone)]
pub struct GpuProfiler {
    inner: Arc<Mutex<Option<ProfilerInner>>>,
}

impl GpuProfiler {
    pub fn new(device: &Device, queue: &Queue) -> Self {
        let features = device.features();
        let has_timestamps = features.contains(wgpu::Features::TIMESTAMP_QUERY);

        let inner = if has_timestamps {
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("gpu_profiler_query_set"),
                ty: wgpu::QueryType::Timestamp,
                count: MAX_QUERIES,
            });

            let resolve_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("gpu_profiler_resolve_buffer"),
                size: (MAX_QUERIES as u64) * std::mem::size_of::<u64>() as u64,
                usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let read_buffer = device.create_buffer(&BufferDescriptor {
                label: Some("gpu_profiler_read_buffer"),
                size: (MAX_QUERIES as u64) * std::mem::size_of::<u64>() as u64,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let timestamp_period = queue.get_timestamp_period();

            Some(ProfilerInner {
                query_set,
                resolve_buffer,
                read_buffer,
                timestamp_period,
                query_names: Vec::with_capacity(MAX_QUERIES as usize),
                query_count: 0,
                enabled: false,
            })
        } else {
            log::warn!("GPU timestamp queries not supported on this device");
            None
        };

        Self {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    pub fn is_available(&self) -> bool {
        self.inner.lock().unwrap().is_some()
    }

    pub fn set_enabled(&self, enabled: bool) {
        if let Some(ref mut inner) = *self.inner.lock().unwrap() {
            inner.enabled = enabled;
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.inner
            .lock()
            .unwrap()
            .as_ref()
            .map(|i| i.enabled)
            .unwrap_or(false)
    }

    pub fn reset(&self) {
        if let Some(ref mut inner) = *self.inner.lock().unwrap() {
            inner.query_names.clear();
            inner.query_count = 0;
        }
    }

    pub fn begin_query(&self, encoder: &mut CommandEncoder, name: &str) -> Option<u32> {
        let mut guard = self.inner.lock().unwrap();
        let inner = guard.as_mut()?;

        if !inner.enabled {
            return None;
        }

        if inner.query_count >= MAX_QUERIES - 1 {
            log::warn!("GPU profiler query limit reached");
            return None;
        }

        let query_index = inner.query_count;
        encoder.write_timestamp(&inner.query_set, query_index);
        inner.query_names.push(name.to_string());
        inner.query_count += 1;

        Some(query_index)
    }

    pub fn end_query(&self, encoder: &mut CommandEncoder, name: &str) -> Option<u32> {
        let mut guard = self.inner.lock().unwrap();
        let inner = guard.as_mut()?;

        if !inner.enabled {
            return None;
        }

        if inner.query_count >= MAX_QUERIES {
            return None;
        }

        let query_index = inner.query_count;
        encoder.write_timestamp(&inner.query_set, query_index);
        inner.query_names.push(format!("{}_end", name));
        inner.query_count += 1;

        Some(query_index)
    }

    pub fn resolve(&self, encoder: &mut CommandEncoder) {
        let guard = self.inner.lock().unwrap();
        if let Some(ref inner) = *guard {
            if inner.enabled && inner.query_count > 0 {
                encoder.resolve_query_set(
                    &inner.query_set,
                    0..inner.query_count,
                    &inner.resolve_buffer,
                    0,
                );
                encoder.copy_buffer_to_buffer(
                    &inner.resolve_buffer,
                    0,
                    &inner.read_buffer,
                    0,
                    (inner.query_count as u64) * std::mem::size_of::<u64>() as u64,
                );
            }
        }
    }

    pub async fn read_results(&self, device: &Device) -> Vec<TimingResult> {
        let (query_count, timestamp_period, query_names) = {
            let guard = self.inner.lock().unwrap();
            match &*guard {
                Some(inner) if inner.enabled && inner.query_count > 0 => (
                    inner.query_count,
                    inner.timestamp_period,
                    inner.query_names.clone(),
                ),
                _ => return vec![],
            }
        };

        let read_buffer = {
            let guard = self.inner.lock().unwrap();
            match &*guard {
                Some(inner) => inner.read_buffer.clone(),
                None => return vec![],
            }
        };

        let slice = read_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });

        _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        if receiver.recv().ok().and_then(|r| r.ok()).is_none() {
            return vec![];
        }

        let timestamps: Vec<u64> = {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice(&data)[..query_count as usize].to_vec()
        };
        read_buffer.unmap();

        let mut results = Vec::new();
        let mut i = 0;
        while i + 1 < timestamps.len() {
            let name = &query_names[i];
            if !name.ends_with("_end")
                && i + 1 < query_names.len()
                && query_names[i + 1].ends_with("_end")
            {
                let start = timestamps[i];
                let end = timestamps[i + 1];
                let duration_ns =
                    ((end.saturating_sub(start)) as f64 * timestamp_period as f64) as u64;
                results.push(TimingResult {
                    name: name.clone(),
                    duration_ns,
                });
                i += 2;
            } else {
                i += 1;
            }
        }

        results
    }

    pub fn aggregate_results(results: &[TimingResult]) -> HashMap<String, (u64, usize)> {
        let mut aggregated: HashMap<String, (u64, usize)> = HashMap::new();
        for result in results {
            let entry = aggregated.entry(result.name.clone()).or_insert((0, 0));
            entry.0 += result.duration_ns;
            entry.1 += 1;
        }
        aggregated
    }

    pub fn print_results(results: &[TimingResult]) {
        let aggregated = Self::aggregate_results(results);
        let mut sorted: Vec<_> = aggregated.into_iter().collect();
        sorted.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));

        let total_ns: u64 = sorted.iter().map(|(_, (ns, _))| ns).sum();

        println!("\n=== GPU Profiling Results ===");
        println!(
            "{:<40} {:>12} {:>8} {:>10}",
            "Operation", "Time (ms)", "Count", "% Total"
        );
        println!("{}", "-".repeat(74));

        for (name, (total_time_ns, count)) in &sorted {
            let time_ms = *total_time_ns as f64 / 1_000_000.0;
            let percent = (*total_time_ns as f64 / total_ns as f64) * 100.0;
            println!(
                "{:<40} {:>12.3} {:>8} {:>9.1}%",
                name, time_ms, count, percent
            );
        }

        println!("{}", "-".repeat(74));
        println!("{:<40} {:>12.3}", "Total", total_ns as f64 / 1_000_000.0);
    }
}

impl Clone for ProfilerInner {
    fn clone(&self) -> Self {
        panic!("ProfilerInner should not be cloned directly")
    }
}
