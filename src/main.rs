#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![feature(test)]
extern crate test;

#[macro_use]
extern crate structopt;
extern crate gag;
extern crate rand;
extern crate serde;
extern crate time;
#[macro_use]
extern crate serde_derive;
extern crate bincode;
extern crate ctrlc;
extern crate plotlib;
extern crate promptly;

mod bench;
mod clioptions;
mod recovkey;

use clioptions::*;

fn main() {
    let opt = clioptions::parse_args();

    match opt.command {
        Command::GenData(gendataopt) => match gendataopt.subcommand {
            GenDataCommand::FindFG(opt) => {
                recovkey::findfg::findfg(opt);
            }
        },
        Command::PostProcess(processopt) => match processopt.subcommand {
            PostProcessCommand::FindFG(opt) => recovkey::processfindfg::process_findfg(opt),
        },
        Command::Bench(benchopt) => match benchopt.subcommand {
            BenchCommand::FindFG(mut opt) => {
                let range = benchopt.min_threads..=benchopt.max_threads;
                bench::bench(
                    |nthreads| {
                        let mut opt = opt.clone();
                        opt.threads = nthreads;
                        recovkey::findfg::findfg(opt) as f64
                    },
                    range,
                );
            }
        },
    }
}
