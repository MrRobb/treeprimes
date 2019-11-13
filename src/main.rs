#[macro_use]
extern crate clap;

#[cfg(feature = "graphviz")]
extern crate dot;

extern crate indicatif;
extern crate itertools;
extern crate primal;
extern crate rand;
extern crate rayon;

use std::error::Error;
use clap::App;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use itertools::Itertools;
use rand::prelude::*;
use rand::seq::IteratorRandom;
use rand::Rng;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::fmt::Display;
use std::mem::swap;

const DEFAULT_N_GENERATIONS: usize = 100;

////////////////////////////////////////////////////////////////////////////////
/// RANDOM
////////////////////////////////////////////////////////////////////////////////

struct ParallelRng {
	streams: Vec<StdRng>,
}

impl ParallelRng {
	fn new(n_streams: usize, seed: u64) -> Self {
		Self {
			streams: (0..n_streams).map(|i| StdRng::seed_from_u64(seed + i as u64)).collect(),
		}
	}

	fn get_stream(&mut self, i: usize) -> &mut StdRng {
		&mut self.streams[i]
	}
}

////////////////////////////////////////////////////////////////////////////////
/// OPERATION
////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, PartialEq, Eq)]
enum Operation {
	X,
	Constant(i128),
	Add,
	Multiply,
}

impl Operation {
	fn eval(&self, values: &[i128], x: i128) -> i128 {
		match &self {
			Self::X => x,
			Self::Constant(n) => *n,
			Self::Add => values.iter().sum(),
			Self::Multiply => values.iter().product(),
		}
	}

	fn arity(&self) -> usize {
		match self {
			Self::X | Self::Constant(_) => 0,
			Self::Add | Self::Multiply => 2,
		}
	}

	fn random(min_depth: usize, min_const: i128, max_const: i128, rng: &mut StdRng) -> Self {
		let operations = vec![
			Self::X,
			Self::Constant(rng.gen_range(min_const, max_const)),
			Self::Add,
			Self::Multiply,
		];

		operations
			.into_iter()
			.filter(|op| match min_depth {
				0 => panic!("The minimum value of depth is 1"),
				1 => op.arity() == 0,
				_ => op.arity() > 0,
			})
			.choose::<StdRng>(rng)
			.unwrap()
	}
}

impl From<String> for Operation {
	fn from(s: String) -> Self {
		match s.as_str() {
			"x" => Self::X,
			"*" => Self::Multiply,
			"+" => Self::Add,
			n => Self::Constant(n.parse().unwrap()),
		}
	}
}

impl ToString for Operation {
	fn to_string(&self) -> String {
		match &self {
			Self::X => "x".to_string(),
			Self::Constant(n) => n.to_string(),
			Self::Add => "+".to_string(),
			Self::Multiply => "*".to_string(),
		}
	}
}

impl Default for Operation {
	fn default() -> Self {
		Self::Constant(0)
	}
}

////////////////////////////////////////////////////////////////////////////////
/// POLYNOMIAL
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, PartialEq, Default)]
struct Polynomial {
	tree: Tree,
	fitness: i64,
}

////////////////////////////////////////////////////////////////////////////////
/// TREE
////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, PartialEq, Default)]
struct Tree {
	root: Operation,
	leaves: Vec<Tree>,
}

impl Tree {
	fn new(op: Operation) -> Self {
		Self {
			root: op,
			leaves: vec![],
		}
	}

	fn eval(&self, x: i128) -> i128 {
		self.root.eval(
			&self.leaves.iter().map(|node| node.eval(x)).collect::<Vec<i128>>(),
			x,
		)
	}

	fn random(min_depth: usize, max_depth: usize, min: i128, max: i128, rng: &mut StdRng) -> Self {
		let root = Operation::random(min_depth, min, max, rng);
		Self {
			leaves: (0..root.arity())
				.map(|_| Self::random(min_depth - 1, max_depth - 1, min, max, rng))
				.collect(),
			root,
		}
	}

	fn size(&self) -> usize {
		1_usize + self.leaves.iter().map(Self::size).sum::<usize>()
	}

	fn depth(&self) -> usize {
		1 + self.leaves.iter().map(Self::depth).max().unwrap_or(0)
	}

	fn mutate(&mut self, p_mutation: f64, min: i128, max: i128, rng: &mut StdRng) {
		if rng.gen_bool(p_mutation) {
			*self = Self::random(self.depth(), self.depth(), min, max, rng);
		}
		else {
			for child in &mut self.leaves {
				child.mutate(p_mutation, min, max, rng);
			}
		}
	}

	fn chop(&mut self, depth: usize, min_const: i128, max_const: i128, rng: &mut StdRng) {
		if depth == 1 {
			self.leaves.clear();
			if self.root.arity() > 0 {
				self.root = Operation::random(1, min_const, max_const, rng)
			}
		}
		else {
			for child in &mut self.leaves {
				child.chop(depth - 1, min_const, max_const, rng);
			}
		}
	}

	#[cfg(feature = "graphviz")]
	fn select_operation(&self, mut index: usize) -> Operation {
		for child in &self.leaves {
			let sz = child.size();
			if sz >= index {
				return child.select_operation(index);
			}
			index -= sz;
		}
		self.root.clone()
	}

	fn merge(op: Operation, leaves: Vec<Self>) -> Self {
		Self { root: op, leaves }
	}

	fn crossover(mom: &mut Self, dad: &mut Self, mut index1: usize, mut index2: usize) {
		for child in &mut mom.leaves {
			let sz = child.size();
			if sz >= index1 {
				Self::crossover(child, dad, index1, index2);
				return;
			}
			index1 -= sz;
		}
		for child in &mut dad.leaves {
			let sz = child.size();
			if sz >= index2 {
				Self::crossover(mom, child, index1, index2);
				return;
			}
			index2 -= sz;
		}
		swap(mom, dad);
	}

	fn parse(s: &str) -> Self {
		// Tokenize
		let tokens = s.chars().filter(|c| !c.is_whitespace());
		let mut output: Vec<String> = Vec::new();
		let mut stack: Vec<String> = Vec::new();
		let mut num: String = String::new();
		let mut neg = true;

		for tok in tokens {
			if tok == 'x' || tok.is_numeric() {
				num.push(tok);
				neg = false;
			}
			else {
				if tok == '-' && neg {
					num.push('-');
					neg = false;
					continue;
				}
				if !num.is_empty() {
					output.push(num.clone());
					num.clear();
				}
				match tok {
					'(' => {
						stack.push("(".to_string());
						neg = true;
					},
					')' => {
						while let Some(v) = stack.pop() {
							if v == "(" {
								break;
							}
							assert_ne!(v, ")");
							output.push(v);
						}
					},
					op => {
						stack.push(op.to_string());
						neg = true;
					},
				}
			}
		}

		if !num.is_empty() {
			output.push(num);
		}

		while let Some(v) = stack.pop() {
			output.push(v);
		}

		let mut tree_stack = Vec::new();

		for op in output {
			let root = Operation::from(op);
			let tree = if root.arity() == 0 {
				Self::new(root)
			}
			else {
				let mut leaves: Vec<Self> = (0..root.arity()).map(|_| tree_stack.pop().unwrap()).collect();

				leaves.reverse();
				Self::merge(root, leaves)
			};

			tree_stack.push(tree);
		}

		assert_eq!(tree_stack.len(), 1);
		tree_stack.last().expect("Error parsing expression").to_owned()
	}

	#[cfg(feature = "graphviz")]
	fn save(&self, filename: String) {
		let mut f = std::fs::File::create(filename).unwrap();
		dot::render(self, &mut f).unwrap();
	}
}

impl ToString for Tree {
	fn to_string(&self) -> String {
		match self.root.arity() {
			0 => self.root.to_string(),
			2 => {
				if self.leaves.is_empty() {
					self.root.to_string()
				}
				else if self.leaves.len() == 2 {
					let prefix = self.leaves.get(0).unwrap().to_string();
					let postfix = self.leaves.get(1).unwrap().to_string();
					format!("({} {} {})", prefix, self.root.to_string(), postfix)
				}
				else {
					let mut s = format!("({} to: [ ", self.root.to_string());
					for leaf in &self.leaves {
						s += &(leaf.to_string() + " ");
					}
					s += "])";
					s
				}
			},
			_ => unreachable!(),
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
/// .DOT GRAPH
////////////////////////////////////////////////////////////////////////////////

#[cfg(feature = "graphviz")]
type Nd = usize;

#[cfg(feature = "graphviz")]
type Ed = (usize, usize);

#[cfg(feature = "graphviz")]
impl<'a> dot::Labeller<'a, Nd, Ed> for Tree {
	fn graph_id(&'a self) -> dot::Id<'a> {
		dot::Id::new("_").unwrap()
	}

	fn node_id(&'a self, n: &Nd) -> dot::Id<'a> {
		dot::Id::new(format!("n{}", *n)).unwrap()
	}

	fn node_label(&'a self, n: &Nd) -> dot::LabelText<'a> {
		let label = self.select_operation(*n).to_string();
		dot::LabelText::label(label)
	}
}

#[cfg(feature = "graphviz")]
impl<'a> dot::GraphWalk<'a, Nd, Ed> for Tree {
	fn nodes(&self) -> dot::Nodes<'a, Nd> {
		let nodes = (1..=self.size()).collect();
		std::borrow::Cow::Owned(nodes)
	}

	fn edges(&'a self) -> dot::Edges<'a, Ed> {
		let mut edges: Vec<Ed> = vec![];
		let myid = self.size();
		let mut offset = 0;

		for child in &self.leaves {
			edges.extend(child.edges().iter().map(|(n1, n2)| (n1 + offset, n2 + offset)));
			offset += child.size();
			edges.push((myid, offset));
		}

		std::borrow::Cow::Owned(edges)
	}

	fn source(&self, e: &Ed) -> Nd {
		e.0
	}

	fn target(&self, e: &Ed) -> Nd {
		e.1
	}
}

////////////////////////////////////////////////////////////////////////////////
/// CRITERION
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
enum Criterion {
	Generations(usize),
	Until(i64),
	TryUntil(usize, i64),
}

impl Criterion {
	fn eval(&self, generation: usize, best_fitness: i64) -> bool {
		match self {
			Self::Generations(g) => generation < *g,
			Self::Until(f) => best_fitness < *f,
			Self::TryUntil(g, f) => generation < *g && best_fitness < *f,
		}
	}

	fn add_generations(self, generations: usize) -> Self {
		match self {
			Self::Generations(_) => Self::Generations(generations),
			Self::Until(f) | Self::TryUntil(_, f) => Self::TryUntil(generations, f),
		}
	}

	fn new_bar(&self) -> ProgressBar {
		let style = ProgressStyle::default_bar()
			.template("{msg} {wide_bar:.cyan/blue} {spinner} Elapsed: {elapsed} ETA: {eta}");

		let progress = match self {
			Self::Generations(g) | Self::TryUntil(g, _) => ProgressBar::new(*g as u64),
			Self::Until(f) => ProgressBar::new(*f as u64),
		};

		match self {
			Self::Generations(g) => {
				if *g >= 100_000 {
					progress.set_draw_delta(*g as u64 / 300);
				}
			},
			Self::TryUntil(g, _) => {
				if *g >= 100_000 {
					progress.set_draw_delta(*g as u64 / 300);
				}
			},
			Self::Until(_) => {},
		}

		progress.set_style(style);
		progress.set_message(&format!("Gens: {} Fitness: {}", 0, 0));
		progress
	}

	fn progress(&self, progress: &ProgressBar, generation: usize, best_fitness: i64) {
		match self {
			Self::Generations(_) => progress.inc(1),
			Self::Until(_) => progress.set_position(best_fitness as u64),
			Self::TryUntil(_, _) => progress.inc(1),
		}
		progress.set_message(&format!("Gens: {} Fitness: {}", generation, best_fitness));
	}
}

impl Default for Criterion {
	fn default() -> Self {
		Self::Generations(DEFAULT_N_GENERATIONS)
	}
}

////////////////////////////////////////////////////////////////////////////////
/// FITNESS
////////////////////////////////////////////////////////////////////////////////

#[derive(Clone)]
enum Fitness {
	CountInInterval,
	From0Consecutive,
	MaxInARow,
	Scalarisation
}

impl Default for Fitness {
	fn default() -> Self {
		Self::From0Consecutive
	}
}

impl Fitness {
	fn eval(&self, ind: &Tree, x_max: usize) -> i64 {
		match self {
			Self::CountInInterval => fitness1(ind, x_max),
			Self::From0Consecutive => fitness2(ind, x_max),
			Self::MaxInARow => fitness3(ind, x_max),
			Self::Scalarisation => fitness4(ind, x_max),
		}
	}

	fn from<E: Error>(id: Result<String, E>) -> Self {
		match id {
			Ok(valid_id) => {
				match valid_id.as_str() {
					"consecutive" => Self::From0Consecutive,
					"interval" => Self::MaxInARow,
					"count" => Self::CountInInterval,
					"scalarisation" => Self::Scalarisation,
					_ => unreachable!()
				}
			}
			Err(_) => Fitness::default()
		}
	}
}

fn fitness1(tree: &Tree, x_max: usize) -> i64 {
	(0..x_max)
		.map(|x| tree.eval(x as i128))
		.filter(|&y| primal::is_prime(y.abs() as u64))
		.unique()
		.count() as i64
}

fn fitness2(tree: &Tree, x_max: usize) -> i64 {
	(0..x_max)
		.map(|x| tree.eval(x as i128))
		.take_while(|&y| primal::is_prime(y.abs() as u64))
		.unique()
		.count() as i64
}

fn fitness3(tree: &Tree, x_max: usize) -> i64 {
	let mut f_max = 0_i64;
	let mut v = vec![];
	for x in 0..x_max {
		let y = tree.eval(x as i128);
		if primal::is_prime(y.abs() as u64) {
			v.push(y)
		}
		else {
			f_max = f_max.max(v.iter().unique().count() as i64);
			v.clear();
		}
	}
	f_max.max(v.iter().unique().count() as i64)
}

fn fitness4(tree: &Tree, x_max: usize) -> i64 {
	let f1 = fitness1(tree, x_max);
	let f2 = fitness2(tree, x_max);
	let f3 = fitness3(tree, x_max);
	f1 + f2 + f3
}

////////////////////////////////////////////////////////////////////////////////
/// SELECTION
////////////////////////////////////////////////////////////////////////////////

fn selection(population: &[Polynomial], mut rng: &mut StdRng) -> Polynomial {
	population
		.choose_multiple::<StdRng>(&mut rng, 5)
		.max_by_key(|ind| ind.fitness)
		.unwrap()
		.clone()
}

////////////////////////////////////////////////////////////////////////////////
/// CROSSOVER
////////////////////////////////////////////////////////////////////////////////

fn crossover(mom: &mut Polynomial, dad: &mut Polynomial, p_crossover: f64, rng: &mut StdRng) {
	if rng.gen_bool(p_crossover) {
		let index1 = rng.gen_range(0, mom.tree.size());
		let index2 = rng.gen_range(0, dad.tree.size());
		Tree::crossover(&mut mom.tree, &mut dad.tree, index1, index2);
	}
}

fn chop(ind: &mut Polynomial, depth: usize, min_const: i128, max_const: i128, rng: &mut StdRng) {
	ind.tree.chop(depth, min_const, max_const, rng);
}

////////////////////////////////////////////////////////////////////////////////
/// MUTATION
////////////////////////////////////////////////////////////////////////////////

fn mutate(ind: &mut Polynomial, p_mutation: f64, min: i128, max: i128, rng: &mut StdRng) {
	ind.tree.mutate(p_mutation, min, max, rng);
}

////////////////////////////////////////////////////////////////////////////////
/// ALGORITHM
////////////////////////////////////////////////////////////////////////////////

fn algorithm(config: &Config) -> Polynomial {
	let mut generation = 0;
	let mut prng = ParallelRng::new(config.n_population, config.seed);

	// Initial population
	let mut population: Vec<Polynomial> = (0..config.n_population)
		.map(|i| Polynomial {
			tree: Tree::random(
				config.min_tree_depth,
				config.max_tree_depth,
				config.min_value,
				config.max_value,
				prng.get_stream(i),
			),
			fitness: 0,
		})
		.collect();

	// Fitness
	population
		.par_iter_mut()
		.for_each(|ind| ind.fitness = config.f_fitness.eval(&ind.tree, config.x_eval_max));

	population.par_sort_by_key(|x| Reverse(x.fitness));

	// Pick best individual
	let mut rank: Vec<Polynomial> = population
		.iter()
		.unique_by(|x| x.fitness)
		.take(config.rank_size)
		.cloned()
		.collect();

	let mut best: &Polynomial = rank.first().unwrap();

	let progress = config.stop_criterion.new_bar();

	while config.stop_criterion.eval(generation, best.fitness) {
		population = prng
			.streams
			.par_iter_mut()
			.chunks(2)
			.map(|mut rng| {
				// Selection
				let mut son = selection(&population, &mut rng[0]);
				let mut daughter = selection(&population, &mut rng[1]);

				// Crossover
				crossover(&mut son, &mut daughter, config.p_crossover, &mut rng[0]);

				// Mutation
				mutate(
					&mut son,
					config.p_mutation,
					config.min_value,
					config.max_value,
					&mut rng[0],
				);
				mutate(
					&mut daughter,
					config.p_mutation,
					config.min_value,
					config.max_value,
					&mut rng[1],
				);

				// Chop
				chop(
					&mut son,
					config.max_tree_depth,
					config.min_value,
					config.max_value,
					&mut rng[0],
				);
				chop(
					&mut daughter,
					config.max_tree_depth,
					config.min_value,
					config.max_value,
					&mut rng[0],
				);

				vec![son, daughter]
			})
			.flatten()
			.collect();

		// Fitness
		population
			.par_iter_mut()
			.for_each(|ind| ind.fitness = config.f_fitness.eval(&ind.tree, config.x_eval_max));

		// Pick best individual
		let worst = rank.last().unwrap().clone();
		let subset: Vec<Polynomial> = population
			.par_iter()
			.filter(|ind| ind.fitness > worst.fitness && !rank.contains(ind))
			.cloned()
			.collect();

		rank.extend(subset.into_iter());
		rank.par_sort_by_key(|x| Reverse(x.fitness));
		rank.resize(config.rank_size, Polynomial::default());
		best = rank.first().unwrap();

		if config.print_every_gen {
			let personal_best = population.par_iter().max_by_key(|ind| ind.fitness).unwrap();

			progress.println(format!(
				"Generation: {} -> Personal: {:?} with {} & Global: {:?} with {}",
				generation,
				personal_best.tree.to_string(),
				personal_best.fitness,
				best.tree.to_string(),
				best.fitness
			));
		}

		config.stop_criterion.progress(&progress, generation, best.fitness);
		generation += 1;
	}

	progress.finish_with_message(&format!(
		"Done with {} generations and fitness {}",
		generation, best.fitness
	));

	if config.print_rank {
		println!();
		println!("---------------------- RANK ----------------------");
		for (i, ind) in rank.iter().enumerate() {
			println!("{} -> {:?} with {}", i, ind.tree.to_string(), ind.fitness);
		}
	}

	#[cfg(feature = "graphviz")]
	best.tree.save("best.dot".into());

	best.clone()
}

////////////////////////////////////////////////////////////////////////////////
/// CONFIG
////////////////////////////////////////////////////////////////////////////////

struct Config {
	min_tree_depth: usize,
	max_tree_depth: usize,
	n_population: usize,
	stop_criterion: Criterion,
	p_crossover: f64,
	p_mutation: f64,
	f_fitness: Fitness,
	x_eval_max: usize,
	min_value: i128,
	max_value: i128,
	rank_size: usize,
	print_rank: bool,
	print_every_gen: bool,
	seed: u64,
}

impl Display for Config {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		writeln!(f, "\nConfiguration of the Algorithm")?;
		writeln!(f, " * Threads: {}", rayon::current_num_threads())?;
		writeln!(f, " * Seed: {}", self.seed)?;
		writeln!(f, " * Population size: {}", self.n_population)?;
		writeln!(f, " * Depth: [{}, {}]", self.min_tree_depth, self.max_tree_depth)?;
		writeln!(f, " * Domain of X (polynomial eval.): [0, {}]", self.x_eval_max)?;
		writeln!(f, " * Domain of constants: [{}, {}]", self.min_value, self.max_value)?;
		writeln!(f, " * Probability of Crossover: {}", self.p_crossover)?;
		writeln!(f, " * Probability of Mutation: {}", self.p_mutation)?;
		writeln!(f, " * Fitness: ...")
	}
}

////////////////////////////////////////////////////////////////////////////////
/// MAIN
////////////////////////////////////////////////////////////////////////////////

fn main() {
	let yml = load_yaml!("../app.yml");
	let app = App::from(yml).get_matches();

	let mut criterion = Criterion::default();
	if app.is_present("target_fitness") {
		criterion = Criterion::Until(value_t_or_exit!(app, "target_fitness", i64))
	}
	if app.is_present("generations") {
		criterion = criterion.add_generations(value_t_or_exit!(app, "generations", usize))
	}

	let fitness = Fitness::from(value_t!(app, "fitness_function", String));

	let config = Config {
		n_population: value_t_or_exit!(app, "population", usize),
		min_tree_depth: value_t_or_exit!(app, "min_depth", usize),
		max_tree_depth: value_t_or_exit!(app, "max_depth", usize),
		stop_criterion: criterion,
		p_crossover: value_t_or_exit!(app, "crossover", f64),
		p_mutation: value_t_or_exit!(app, "mutation", f64),
		f_fitness: fitness,
		x_eval_max: value_t_or_exit!(app, "max_x", usize),
		min_value: value_t_or_exit!(app, "min_value", i128),
		max_value: value_t_or_exit!(app, "max_value", i128),
		rank_size: value_t_or_exit!(app, "rank_size", usize),
		print_rank: !app.is_present("hide-rank"),
		print_every_gen: app.is_present("verbose"),
		seed: value_t!(app, "seed", u64).unwrap_or_else(|_| rand::random()),
	};

	println!("{}", config);

	if let Some(matches) = app.subcommand_matches("eval") {
		let input = value_t_or_exit!(matches, "input", String);
		let tree = Tree::parse(&input);
		println!("{}\n", tree.to_string());

		for x in 0..config.x_eval_max {
			let y = tree.eval(x as i128);
			println!(
				"x = {} -> y = {} (is_prime: {})",
				x,
				y,
				primal::is_prime(y.abs() as u64)
			);
		}
	}
	else {
		let _ = algorithm(&config);
	}
}
