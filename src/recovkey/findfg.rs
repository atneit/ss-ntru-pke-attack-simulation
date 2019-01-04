use super::processfindfg::{calc_error_rate, calc_hueristic_coeff, ResultsV2, SampleV1};
use super::{FLOAT, INT, LONG};
use bincode;
use ctrlc;
use rand;
use rand::distributions::Normal;
use rand::prelude::Distribution;
use std::cmp::min;
use std::fs::File;
use std::io::Write;
use std::sync::mpsc;
use std::sync::mpsc::SendError;
use std::sync::mpsc::SyncSender;
use std::sync::{atomic::AtomicBool, atomic::Ordering, Arc};
use std::thread;

#[derive(StructOpt, Debug, Default, Clone)]
/// Calculates multiple samples which can be used to guess f and g.
pub struct FindFGOptions {
    #[structopt(short = "t", long = "threads", default_value = "4")]
    /// The number of threads uesed to produce samples
    pub threads: u8,

    #[structopt(short = "S", long = "samples", default_value = "1200")]
    /// The number of sample generating trials for each thread
    pub samples: u32,

    #[structopt(short = "N", long = "keylen", default_value = "1024")]
    /// The length of the vectors f,g,r & e
    pub N: u16,

    #[structopt(short = "s", long = "sigma", default_value = "724.0")]
    /// The sigma
    pub sigma: FLOAT,

    #[structopt(short = "F", long = "assumed_f", default_value = "4923.2")]
    /// The assumed large value of f[0] and f[1]. Default: SIGMA * 6.8
    pub assumed_f: FLOAT,

    #[structopt(short = "R", long = "assumed_r", default_value = "6660.8")]
    /// The assumed large value of r[0] and r[1]. Default: SIGMA * 9.2
    pub assumed_r: FLOAT,

    #[structopt(short = "q", long = "mod_q", default_value = "671088640")] //2^29 + 2^27
    /// A sample is only valid if the following condition hold true: r * f + e * g >= mod_q / 4.
    /// The default value is 2^29 + 2^27
    pub mod_q: LONG,

    #[structopt(short = "v", long = "view", default_value = "15")]
    /// Specifies 'the view' i.e. how many positions (excluding the
    /// first two) to display and calculate the aggregate of
    pub view: usize,

    #[structopt(short = "o", long = "outfile")]
    /// Specifies the path of where to store the output
    pub outfile: Option<String>,
}

#[derive(Debug, Clone)]
struct ThreadContext {
    options: FindFGOptions,
    thread_id: u8,
    tx: SyncSender<ThreadMsg>,
    f: Vec<INT>,
    g: Vec<INT>,
}

struct ThreadMsg {
    sender: u8,
    msg: MsgKind,
}

enum MsgKind {
    FoundSample(SampleV1),
    Exiting(LONG),
}

fn gen_pair(sigma: f64, N: u16, first: FLOAT) -> impl Iterator<Item = (INT, INT)> {
    let D = Normal::new(0.0, sigma);
    (0..N)
        .map(move |i| {
            // "assume" f has two big entries in the beginning
            let f = if i <= 1 {
                first
            } else {
                D.sample(&mut rand::thread_rng())
            };
            let g = D.sample(&mut rand::thread_rng());
            (f, g)
        }).map(|(f, g)| (f.round() as INT, g.round() as INT))
}

fn gen_sample(opt: &FindFGOptions, f: &Vec<INT>, g: &Vec<INT>, trial: LONG) -> SampleV1 {
    let empty_sample = SampleV1 {
        r: Vec::with_capacity(opt.N as usize),
        e: Vec::with_capacity(opt.N as usize),
        sum: 0,
        trial,
    };
    //produces iterator (length N) of (r, e) tuples
    let sample = gen_pair(opt.sigma, opt.N, opt.assumed_r)
             //produces iterator of ((r, e), (f, g))
            .zip(f.iter().zip(g.iter()))
            // calculates sum s
            .fold(empty_sample, |mut sample, ((r, e), (f, g))| {
                sample.r.push(r);
                sample.e.push(e);
                sample.sum += r as LONG * (*f) as LONG + e  as LONG * (*g) as LONG;
                sample
            });
    sample
}

fn find_samples(
    ctx: ThreadContext,
    worker_exit_condition: Arc<AtomicBool>,
) -> Result<(), SendError<ThreadMsg>> {
    for i in 0.. {
        let sample = gen_sample(&ctx.options, &ctx.f, &ctx.g, i);
        if sample.sum >= ctx.options.mod_q / 4 {
            // If the reciever is closed, we simply close down the thread
            ctx.tx.send(ThreadMsg {
                sender: ctx.thread_id,
                msg: MsgKind::FoundSample(sample),
            })?;
        }
        if worker_exit_condition.load(Ordering::Relaxed) {
            ctx.tx.send(ThreadMsg {
                sender: ctx.thread_id,
                msg: MsgKind::Exiting(i),
            })?;
            return Ok(());
        }
    }
    unreachable!();
}

pub fn findfg(options: FindFGOptions) -> FLOAT {
    // Open file, if specified
    let outfile = if let Some(ref outpath) = options.outfile {
        Some(File::create(outpath).expect("cannot open outfile for writing!"))
    } else {
        None
    };

    // generate test key (f, g)
    let (f, g): (Vec<INT>, Vec<INT>) =
        gen_pair(options.sigma, options.N, options.assumed_f).unzip();
    let fcopy = f.clone();
    let gcopy = g.clone();

    let mut results = ResultsV2 {
        all_samples: Vec::new(),
        total_trials: 0,
        f: fcopy,
        g: gcopy,
        assumed_f: options.assumed_f,
        assumed_r: options.assumed_r,
        mod_q: options.mod_q as INT,
        sigma: options.sigma,
    };

    let worker_exit_condition = Arc::new(AtomicBool::new(false));

    {
        let worker_exit_condition = worker_exit_condition.clone();
        // set up ctrl-c exit handler
        // we ignore errors, since we may be calling this function multiple times
        let _ = ctrlc::set_handler(move || {
            println!("\t<=== CTRL-C detected, telling workers to stop! ===>");
            worker_exit_condition.store(true, Ordering::Relaxed);
        });
    }

    let (rx, handles) = {
        // Create communication channel for thread work.
        // We use sync channel so we can detect immediately
        // when no one is receiveing any more
        let (tx, rx) = mpsc::sync_channel(0);

        let ctx = ThreadContext {
            options: options.clone(),
            thread_id: 0,
            tx,
            f,
            g,
        };

        //Spawn threads to find samples
        let handles: Vec<_> = (0..options.threads)
            .map(|t| {
                let mut ctx = ctx.clone();
                ctx.thread_id = t;
                let worker_exit_condition = worker_exit_condition.clone();
                thread::spawn(move || {
                    let _ = find_samples(ctx, worker_exit_condition);
                })
            }).collect();

        //we drop tx here, otherwise we won't find out if the threads are finsihed!
        (rx, handles)
    };
    let viewrange = 2..min(options.view + 2, (options.N) as usize);

    let mut total_trials = 0;
    let mut i = 0;

    for msg in rx.iter() {
        match msg.msg {
            MsgKind::FoundSample(received) => {
                i += 1;
                println!(
                    "Sample {}/{}, trial: {}#{}, r: {:?}",
                    i,
                    options.samples,
                    msg.sender,
                    received.trial,
                    &received.r[viewrange.clone()]
                );
                results.all_samples.push(received);
                if i >= options.samples as usize {
                    // Tell the threads to exit, then we just wait for the channel
                    // to close on its own
                    worker_exit_condition.store(true, Ordering::Relaxed);
                }
            }
            MsgKind::Exiting(trials) => {
                println!(
                    "Thread {} stopped working after {} completed trials.",
                    msg.sender, trials
                );
                total_trials += trials;
            }
        };
    }

    print!("Last sample receieved. Status of worker threads... ");
    for handle in handles {
        handle.join().expect("Couldn't join on the worker thread!");
    }
    println!("Closed!");
    results.total_trials = total_trials;

    if let Some(mut outfile) = outfile {
        // Write version number of serialization format
        outfile
            .write(&ResultsV2::magic_prefix_u128())
            .expect("Could not write magic number to file");
        if bincode::serialize_into(outfile, &results).is_err() {
            println!("Could not serialize results to outfile!");
        }
    }

    if results.all_samples.len() > 0 {
        let coeff = calc_hueristic_coeff(
            options.N as INT,
            options.sigma,
            options.mod_q as INT,
            options.assumed_f,
            options.assumed_r,
            0.0,
        );
        let error_rate = calc_error_rate(coeff, &mut results, 0);

        println!(
            "  --- Calculations based on {} samples ---",
            results.all_samples.len()
        );
        println!("    error_rate:      {:?}", error_rate);
    }

    println!("Bye!");

    total_trials as FLOAT
}

#[cfg(test)]
mod tests {
    use super::*;
    use structopt::StructOpt;

    #[test]
    fn test_gen_fg() {
        let (f, _g): (Vec<INT>, Vec<INT>) = gen_pair(724.0, 1024, 4923.2).unzip();
        assert_eq!(f[0], 4923 as INT);
        assert_eq!(f[1], 4923 as INT);
        assert!(f[2] != 4923 as INT); //probably
    }

    extern crate test;
    use test::Bencher;

    #[bench]
    fn bench_gen_fg(b: &mut Bencher) {
        b.iter(|| {
            let (f, g): (Vec<INT>, Vec<INT>) = gen_pair(724.0, 1024, 4923.2).unzip();
            test::black_box(&f);
            test::black_box(&g);
        })
    }

    #[bench]
    fn bench_1_trial(b: &mut Bencher) {
        let (f, g): (Vec<INT>, Vec<INT>) = gen_pair(724.0, 1024, 4923.2).unzip();
        let v: Vec<String> = Vec::new();
        //This will create a struct filled with default values
        let opt = FindFGOptions::from_iter(v.iter());
        b.iter(|| {
            let sample = gen_sample(&opt, &f, &g, 0);
            test::black_box(&sample);
        })
    }
}
