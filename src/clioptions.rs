use recovkey::findfg::FindFGOptions;
use recovkey::processfindfg::ProcessFindFGOptions;
use structopt::StructOpt;

#[derive(StructOpt, Debug)]
pub enum Command {
    /// A collection of expensive data generation subcommands
    #[structopt(name = "datagen")]
    GenData(GenDataOpt),

    /// A collection of less expensive data post
    /// processing commands (generated by 'datagen')
    #[structopt(name = "process")]
    PostProcess(PostProcessOpt),

    /// A collection of benchmarking commands,  usefull for determining
    /// the optimal arguments for some of the 'datagen' subcommands
    #[structopt(name = "bench")]
    Bench(BenchOpt),
}

#[derive(StructOpt, Debug)]
#[structopt(name = "command")]
pub enum GenDataCommand {
    /// Attempts to generate data which is used in finding f and g
    #[structopt(name = "findfg")]
    FindFG(FindFGOptions),
}

#[derive(StructOpt, Debug)]
pub struct GenDataOpt {
    #[structopt(subcommand)]
    pub subcommand: GenDataCommand,
}

#[derive(StructOpt, Debug)]
pub struct PostProcessOpt {
    #[structopt(subcommand)]
    pub subcommand: PostProcessCommand,
}

#[derive(StructOpt, Debug)]
pub enum PostProcessCommand {
    /// Reads the raw data from the 'datagen findfg' and attempts to find f and g
    #[structopt(name = "findfg",)]
    FindFG(ProcessFindFGOptions),
}

#[derive(StructOpt, Debug)]
pub struct BenchOpt {
    #[structopt(subcommand)]
    pub subcommand: BenchCommand,
    #[structopt(short = "n", long = "min_threads", default_value = "1")]
    pub min_threads: u8,
    #[structopt(short = "x", long = "max_threads", default_value = "16")]
    pub max_threads: u8,
}

#[derive(StructOpt, Debug)]
#[structopt(name = "command")]
pub enum BenchCommand {
    /// Benchmarks the 'datagen findfg' command
    #[structopt(name = "datagen-findfg",)]
    FindFG(FindFGOptions),
}

#[derive(StructOpt, Debug)]
pub struct Opt {
    #[structopt(subcommand)]
    pub command: Command,
}

pub fn parse_args() -> Opt {
    Opt::from_args()
}