use gag::Gag;
use std::ops::RangeInclusive;
use time::precise_time_s;

/// Accepts a closure which takes the number of worker threads
/// to spawn as its only argument and return a 'performance indicator'
/// which is a measure of the amount of work performed
/// (e.g. number of iterations or samples or whatever).
pub fn bench<F: Fn(u8) -> f64>(func: F, numthreads: RangeInclusive<u8>) {
    let results: Vec<_> = numthreads
        .clone()
        .map(|nb| {
            println!("Benchmarking with {} worker thread... ", nb);
            let perfomance = {
                // This disables stdout for the duration of the function
                let _print_gag = Gag::stdout().unwrap();
                // start the clock
                let start = precise_time_s();
                // do the thing
                let counter = func(nb);
                //Stop the clock
                let end = precise_time_s();
                let duration = end - start;
                counter / duration
            };
            perfomance
        }).collect();
    println!("\t| threads:\t| performance (calcs/second):\t|");
    for (i, result) in results.iter().enumerate() {
        println!("\t| {}\t\t| {}\t\t|", *numthreads.start() + i as u8, result);
    }
}
