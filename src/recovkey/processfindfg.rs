use super::{FLOAT, INT, LONG};
use bincode;
use promptly::{prompt, prompt_default};
use std::cmp::min;
use std::collections::binary_heap::BinaryHeap;
use std::fs::File;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use structopt;

#[derive(Default, Serialize, Deserialize, Debug)]
pub struct SampleV1 {
    pub e: Vec<INT>,
    pub r: Vec<INT>,
    pub sum: LONG,
    pub trial: LONG,
}

#[derive(Default, Serialize, Deserialize, Debug)]
pub struct SampleV0 {
    pub e: Vec<INT>,
    pub r: Vec<INT>,
    pub sum: LONG,
    pub trial: u32,
}

#[derive(Default, Serialize, Deserialize, Debug)]
pub struct ResultsV2 {
    pub all_samples: Vec<SampleV1>,
    pub f: Vec<INT>,
    pub g: Vec<INT>,
    pub total_trials: LONG,
    pub sigma: FLOAT,
    pub mod_q: INT,
    pub assumed_f: FLOAT,
    pub assumed_r: FLOAT,
}

impl ResultsV2 {
    pub fn magic_prefix_u128() -> Vec<u8> {
        vec![0, 0, 0, 0, 0, 0, 0, 2]
    }
}

#[derive(Default, Serialize, Deserialize, Debug)]
pub struct ResultsV1 {
    pub all_samples: Vec<SampleV1>,
    pub r_sum: Vec<INT>,
    pub e_sum: Vec<INT>,
    pub r_orthogonal: Vec<BinaryHeap<INT>>,
    pub e_orthogonal: Vec<BinaryHeap<INT>>,
    pub total_trials: LONG,
    pub f: Vec<INT>,
    pub g: Vec<INT>,
}

#[derive(Default, Serialize, Deserialize, Debug)]
pub struct ResultsV0 {
    pub all_samples: Vec<SampleV0>,
    pub r_sum: Vec<INT>,
    pub e_sum: Vec<INT>,
    pub r_orthogonal: Vec<BinaryHeap<INT>>,
    pub e_orthogonal: Vec<BinaryHeap<INT>>,
    pub trials: u32,
    pub f: Vec<INT>,
    pub g: Vec<INT>,
}

#[derive(StructOpt, Debug, Default, Clone)]
#[structopt(raw(setting = "structopt::clap::AppSettings::ColoredHelp"))]
/// Calculates multiple samples which can be used to guess f and g.
pub struct ProcessFindFGOptions {
    #[structopt(short = "v", default_value = "15")]
    /// Specifies 'the view' i.e. how many positions (excluding the
    /// first two) to display and calculate the aggregate of. The special value
    /// '0' means display all positions in the vector
    pub view: usize,

    /// Specifies the path of from where to read the output of the
    /// 'findfg' command
    pub infile: Vec<String>,

    #[structopt(short = "N", long = "samples", default_value = "0")]
    /// Specifies the number of samples to base the calculations on.
    /// The number 0 (default) results in all the samples beeing used.
    pub samples: Vec<usize>,

    #[structopt(
        short = "m",
        long = "coeff_mod",
        default_value = "0",
        raw(set = "structopt::clap::ArgSettings::AllowLeadingHyphen")
    )]
    /// If set, the calculated heuristic coefficient is modifier by this
    /// value (using addition)
    pub coeff_mod: FLOAT,

    #[structopt(short = "o", long = "outdir")]
    /// If set, the calculation output (for plotting) is written to this directory
    pub outdir: Option<String>,
}

pub fn calc_error_rate(heuristic_coeff: FLOAT, results: &ResultsV2, samples: usize) -> FLOAT {
    let N = results.f.len() as INT;
    let N2 = N as usize * 2;
    let max_samples = results.all_samples.len();
    let sigma = results.sigma as INT;
    let samples = if samples == 0 {
        max_samples
    } else {
        min(samples, max_samples)
    };

    //Calculate K
    let K: Vec<_> = results.f.iter().chain(results.g.iter()).collect();

    //Calculate V_sum
    let V_sum: Vec<_> = results
        .all_samples
        .iter()
        .take(samples) //restrict number of samples we want to use
        .fold(vec![0; N2], |sum, sample| {
            sample
                .r
                .iter()
                .chain(sample.e.iter())
                .zip(sum)
                .map(|(f, s)| f + s)
                .collect()
        });

    //Calculate V_mean
    let V_mean: Vec<_> = V_sum
        .iter()
        .map(|ri| (*ri as FLOAT / samples as FLOAT).round() as INT)
        .collect();

    // K_hat
    let K_hat: Vec<_> = V_mean
        .iter()
        .map(|v| *v as FLOAT * heuristic_coeff)
        .collect();

    // K_delta
    let K_delta: Vec<_> = K
        .iter()
        .zip(K_hat.iter())
        .skip(2)//Do not calculate over the first two positions
        .map(|(k, k_hat)| k_hat - **k as FLOAT)
        .collect();

    //Finally we calculate the reduction rate
    let sum = K_delta.iter().fold(0.0, |sum, d| {
        let pow = d.powf(2.0);
        let tot = sum + pow;
        tot
    });
    let error = sum.sqrt();
    let error_rate = error / (((N2 - 2) as FLOAT).sqrt() * sigma as FLOAT);

    error_rate
}

pub fn calc_hueristic_coeff(
    N: INT,
    sigma: FLOAT,
    q: INT,
    F: FLOAT,
    R: FLOAT,
    coeff_mod: FLOAT,
) -> FLOAT {
    let C = F * R * 2.0;
    // ((N-2)*sigma^2) / ((q/4)-C)
    (((N * 2 - 2) as FLOAT * sigma.powf(2.0)) / ((q as FLOAT / 4.0) - C)) + coeff_mod
}

fn read_results_v0(inpath: &str) -> ResultsV1 {
    let infile = File::open(inpath).expect("cannot open infile for reading!");
    let mut v0: ResultsV0 =
        bincode::deserialize_from(infile).expect("Input file is of an unrecognised format");
    let v1 = ResultsV1 {
        all_samples: v0
            .all_samples
            .drain(..)
            .map(|s0| SampleV1 {
                e: s0.e,
                r: s0.r,
                sum: s0.sum,
                trial: s0.trial as LONG,
            }).collect(),
        r_sum: v0.r_sum,
        e_sum: v0.e_sum,
        r_orthogonal: v0.r_orthogonal,
        e_orthogonal: v0.e_orthogonal,
        total_trials: v0.trials as LONG,
        f: v0.f,
        g: v0.g,
    };

    v1
}

fn read_results_v1(inpath: &str) -> ResultsV2 {
    // We try again
    let infile = File::open(inpath).expect("cannot open infile for reading!");
    let v1: ResultsV1 = match bincode::deserialize_from(infile) {
        Ok(v1) => v1,
        Err(_) => read_results_v0(inpath),
    };
    let new_res = ResultsV2 {
        all_samples: v1.all_samples,
        f: v1.f,
        g: v1.g,
        total_trials: v1.total_trials,
        sigma: prompt_default(" > missing value for sigma", 724.0),
        mod_q: prompt_default(" > missing value for q", 671088640),
        assumed_f: prompt_default(" > missing value for F", 4923.2),
        assumed_r: prompt_default(" > missing value for R", 6660.8),
    };

    let path: Option<PathBuf> = prompt(" > Enter a path to save the results using the new format");

    if let Some(path) = path {
        let outfile = File::create(path);
        if outfile.is_err() {
            println!("Could not open file for writing!");
        } else {
            let mut outfile = outfile.unwrap();
            outfile
                .write(&ResultsV2::magic_prefix_u128())
                .expect("Could not write magic number to file");
            if bincode::serialize_into(outfile, &new_res).is_err() {
                println!("Could not serialize results to outfile!");
            }
        }
    }

    new_res
}

fn read_results_v2(inpath: &str) -> ResultsV2 {
    println!("Loading {}...", inpath);
    let mut infile = File::open(inpath).expect("cannot open infile for reading!");

    //check for magic number
    let mut buf = ResultsV2::magic_prefix_u128();
    infile
        .read_exact(&mut buf)
        .expect("Cannot read from infile!");

    let results = if buf == ResultsV2::magic_prefix_u128() {
        match bincode::deserialize_from(infile) {
            Ok(results) => results,
            Err(_) => read_results_v1(inpath),
        }
    } else {
        read_results_v1(inpath)
    };

    results
}

fn neg_log2(x: &FLOAT) -> FLOAT {
    0.0 - x.log2()
}

pub fn process_findfg(options: ProcessFindFGOptions) {
    let files: Vec<_> = options
        .infile
        .iter()
        .map(|p| (p, read_results_v2(p)))
        .collect();
    let all_rates: Vec<_> = files
        .iter()
        .map(|(infile, results)| {
            let N = results.f.len() as INT;
            let max_samples = results.all_samples.len();
            let sigma = results.sigma;
            let q = results.mod_q;
            let coeff = calc_hueristic_coeff(
                N,
                sigma,
                q,
                results.assumed_f,
                results.assumed_r,
                options.coeff_mod,
            );

            let failure_rate = results.all_samples.len() as FLOAT / results.total_trials as FLOAT;
            println!("=== Simulation summary ({}) ===", infile);
            println!("  N: {}", N);
            println!("  samples: {}", max_samples);
            println!("  total trials: {}", results.total_trials);
            println!("  failure rate: {}", failure_rate);
            println!("  sigma: {}", sigma);
            println!("  q: {}", q);
            println!("  heuristics coefficient: {}", coeff);

            let rates: Vec<_> = options
                .samples
                .iter()
                .map(|s| {
                    let s = *s;
                    (s, calc_error_rate(coeff, &results, s))
                }).collect();
            (failure_rate, rates)
        }).collect();

    if let Some(outdir) = options.outdir {
        let base = Path::new(&outdir);
        println!("=== LaTeX error_rate vs samples ===");
        for (failure_rate, rates) in &all_rates {
            let fname = format!("err_norm_num_vecs-{:.2}.table", neg_log2(failure_rate));
            let fname = base.clone().join(fname);
            println!("Writing to {:?}", fname);
            let mut file = File::create(fname).expect("Could not create file");
            for (s, rate) in rates {
                writeln!(file, "  {} {}", s, rate);
            }
        }

        println!("=== LaTeX error_rate vs failure_rate ===");
        for i in 0..all_rates.first().unwrap().1.len() {
            let fname = format!("error_norm_err_rate-{}.table", i);
            let fname = base.clone().join(fname);
            println!("Writing to {:?}", fname);
            let mut file = File::create(fname).expect("Could not create file");
            for (failure_rate, rates) in &all_rates {
                writeln!(file, "  {} {}", neg_log2(failure_rate), rates[i].1);
            }
        }

        println!("=== LaTeX failure_rate vs samples vs error_rate ===");
        let fname = format!("err_norm_combined.table");
        let fname = base.clone().join(fname);
        println!("Writing to {:?}", fname);
        let mut file = File::create(fname).expect("Could not create file");
        for (failure_rate, rates) in &all_rates {
            let lograte = neg_log2(failure_rate);
            for (s, rate) in rates {
                writeln!(file, "  {} {} {}", lograte, s, rate);
            }
            writeln!(file, "");
        }
    }
}
