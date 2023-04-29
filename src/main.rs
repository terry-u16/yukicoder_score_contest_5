use std::cmp::Reverse;
use std::io::Write;
use std::{collections::BinaryHeap, fmt::Display};

use grid::{inv, Coordinate, Map2d, ADJACENTS};
use judge::{OnlineJudge, SelfJudge};

use crate::{judge::Judge, rand::Xoshiro256};

pub trait ChangeMinMax {
    fn change_min(&mut self, v: Self) -> bool;
    fn change_max(&mut self, v: Self) -> bool;
}

impl<T: PartialOrd> ChangeMinMax for T {
    fn change_min(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }

    fn change_max(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
}

macro_rules! get {
      ($t:ty) => {
          {
              let mut line: String = String::new();
              std::io::stdin().read_line(&mut line).unwrap();
              line.trim().parse::<$t>().unwrap()
          }
      };
      ($($t:ty),*) => {
          {
              let mut line: String = String::new();
              std::io::stdin().read_line(&mut line).unwrap();
              let mut iter = line.split_whitespace();
              (
                  $(iter.next().unwrap().parse::<$t>().unwrap(),)*
              )
          }
      };
      ($t:ty; $n:expr) => {
          (0..$n).map(|_|
              get!($t)
          ).collect::<Vec<_>>()
      };
      ($($t:ty),*; $n:expr) => {
          (0..$n).map(|_|
              get!($($t),*)
          ).collect::<Vec<_>>()
      };
      ($t:ty ;;) => {
          {
              let mut line: String = String::new();
              std::io::stdin().read_line(&mut line).unwrap();
              line.split_whitespace()
                  .map(|t| t.parse::<$t>().unwrap())
                  .collect::<Vec<_>>()
          }
      };
      ($t:ty ;; $n:expr) => {
          (0..$n).map(|_| get!($t ;;)).collect::<Vec<_>>()
      };
}

#[allow(unused_macros)]
macro_rules! chmin {
    ($base:expr, $($cmps:expr),+ $(,)*) => {{
        let cmp_min = min!($($cmps),+);
        if $base > cmp_min {
            $base = cmp_min;
            true
        } else {
            false
        }
    }};
}

#[allow(unused_macros)]
macro_rules! chmax {
    ($base:expr, $($cmps:expr),+ $(,)*) => {{
        let cmp_max = max!($($cmps),+);
        if $base < cmp_max {
            $base = cmp_max;
            true
        } else {
            false
        }
    }};
}

#[allow(unused_macros)]
macro_rules! min {
    ($a:expr $(,)*) => {{
        $a
    }};
    ($a:expr, $b:expr $(,)*) => {{
        std::cmp::min($a, $b)
    }};
    ($a:expr, $($rest:expr),+ $(,)*) => {{
        std::cmp::min($a, min!($($rest),+))
    }};
}

#[allow(unused_macros)]
macro_rules! max {
    ($a:expr $(,)*) => {{
        $a
    }};
    ($a:expr, $b:expr $(,)*) => {{
        std::cmp::max($a, $b)
    }};
    ($a:expr, $($rest:expr),+ $(,)*) => {{
        std::cmp::max($a, max!($($rest),+))
    }};
}

const N: usize = 14;

#[derive(Debug, Clone)]
struct Input {
    n: usize,
    t: usize,
    people: Vec<Person>,
}

impl Input {
    fn read_input() -> Input {
        let (n, t) = get!(usize, usize);
        let mut people = vec![];

        for _ in 0..n {
            let (r0, c0, r1, c1) = get!(usize, usize, usize, usize);
            people.push(Person::new(
                Coordinate::new(r0 - 1, c0 - 1),
                Coordinate::new(r1 - 1, c1 - 1),
            ));
        }

        Input { n, t, people }
    }
}

#[derive(Debug, Clone, Copy)]
struct Person {
    home: Coordinate,
    company: Coordinate,
}

impl Person {
    fn new(home: Coordinate, company: Coordinate) -> Self {
        Self { home, company }
    }
}

#[derive(Debug, Clone)]
struct State {
    money: i64,
    collaborator: i64,
    map: Map2d<[bool; 4]>,
}

impl State {
    fn init() -> Self {
        let map = Map2d::new(vec![[false; 4]; N * N], N);
        Self {
            money: 1000000,
            collaborator: 1,
            map,
        }
    }

    fn calc_construction_cost(&self) -> i64 {
        for i in 1..100 {
            if i * i == self.collaborator {
                return 10_000_000 / i;
            }
        }

        (1e7 / (self.collaborator as f64).sqrt()).floor() as i64
    }

    fn can_construct(&self) -> bool {
        self.money >= self.calc_construction_cost()
    }
}

enum Action {
    Construct(Coordinate, Coordinate),
    Collaboration,
    Money,
}

impl Action {
    fn apply(&self, state: &mut State) {
        if let &Action::Construct(p, q) = self {
            let mut dir = !0;

            for d in 0..4 {
                let adj = ADJACENTS[d];
                let next = p + adj;

                if next == q {
                    dir = d;
                    break;
                }
            }

            state.map[p][dir] = true;
            state.map[q][inv(dir)] = true;
        }
    }
}

impl Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Construct(p, q) => write!(
                f,
                "1 {} {} {} {}",
                p.row + 1,
                p.col + 1,
                q.row + 1,
                q.col + 1
            ),
            Action::Collaboration => write!(f, "2"),
            Action::Money => write!(f, "3"),
        }
    }
}

fn main() {
    let mut state = State::init();
    let mut judge = get_judge();
    let input = judge.read_input();
    let blueprint = annealing(&input, AnnealingState::new(), 1.5);
    eprintln!("{}", blueprint.count);

    for turn in 0..input.t {
        judge.update_state(&mut state);
        let action = get_action(&input, &state, &blueprint, turn);
        action.apply(&mut state);
        judge.apply(action);
    }
}

fn get_judge() -> Box<dyn Judge> {
    match std::env::var("LOCAL") {
        Ok(_) => Box::new(SelfJudge::new()),
        Err(_) => Box::new(OnlineJudge),
    }
}

fn get_action(input: &Input, state: &State, blueprint: &AnnealingState, turn: usize) -> Action {
    if turn < 70 {
        return Action::Collaboration;
    }

    if turn >= 350 || !state.can_construct() {
        return Action::Money;
    }

    let mut candidates = vec![];

    for row in 0..N {
        for col in 0..N {
            let c = Coordinate::new(row, col);

            for &dir in &[1, 2] {
                if blueprint.map[c][dir] {
                    candidates.push((c, c + ADJACENTS[dir]));
                }
            }
        }
    }

    const CENTER: Coordinate = Coordinate::new(6, 6);
    candidates.sort_unstable();
    candidates.dedup();
    candidates.sort_by_key(|(p, q)| p.dist(&CENTER).min(q.dist(&CENTER)));

    for &(p, q) in &candidates {
        let mut dir = !0;

        for d in 0..4 {
            let adj = ADJACENTS[d];
            let next = p + adj;

            if next == q {
                dir = d;
                break;
            }
        }

        if !state.map[p][dir] {
            return Action::Construct(p, q);
        }
    }

    Action::Money
}

const MAX_HIGHWAY: usize = 70;

#[derive(Debug, Clone)]
struct AnnealingState {
    map: Map2d<[bool; 4]>,
    count: usize,
}

impl AnnealingState {
    fn new() -> Self {
        let mut map = Map2d::new(vec![[false; 4]; N * N], N);

        for &row in &[2, 6, 11] {
            for col in 2..10 {
                let c = Coordinate::new(row, col);
                map[c][1] = true;
                map[c + ADJACENTS[1]][3] = true;
            }
        }

        for &col in &[2, 6, 11] {
            for row in 2..10 {
                let c = Coordinate::new(row, col);
                map[c][2] = true;
                map[c + ADJACENTS[2]][0] = true;
            }
        }

        let mut count = 0;

        for row in 0..N {
            for col in 0..N {
                for &dir in &[1, 2] {
                    if map[Coordinate::new(row, col)][dir] {
                        count += 1;
                    }
                }
            }
        }

        Self { map, count }
    }

    fn calc_score(&self, input: &Input) -> i64 {
        let mut distances = Map2d::new(vec![Map2d::new(vec![], N); N * N], N);
        let mut queue = BinaryHeap::new();

        for row0 in 0..N {
            for col0 in 0..N {
                let mut dists = Map2d::new(vec![std::i32::MAX / 2; N * N], N);
                queue.clear();
                dists[Coordinate::new(row0, col0)] = 0;
                queue.push(Reverse((0, Coordinate::new(row0, col0))));

                while let Some(Reverse((dist, c))) = queue.pop() {
                    if dist > dists[c] {
                        continue;
                    }

                    for dir in 0..4 {
                        let next = c + ADJACENTS[dir];
                        let next_cost = dist + if self.map[c][dir] { 223 } else { 1000 };

                        if next.in_map(N) && dists[next].change_min(next_cost) {
                            queue.push(Reverse((next_cost, next)));
                        }
                    }
                }

                distances[Coordinate::new(row0, col0)] = dists;
            }
        }

        let mut score = 0;

        for &person in &input.people {
            let dist = distances[person.home][person.company];

            for count in 0.. {
                let remain = dist - 223 * count;
                if remain % 1000 == 0 {
                    score += 60 * count as i64;
                    break;
                }
            }
        }

        score
    }

    fn can_flip(&self, target: Coordinate, dir: usize) -> bool {
        let next = target + ADJACENTS[dir];
        next.in_map(N) && (self.count < MAX_HIGHWAY || self.map[target][dir])
    }

    fn flip(&mut self, target: Coordinate, dir: usize) {
        let next = target + ADJACENTS[dir];
        if self.map[target][dir] {
            self.count -= 1;
        } else {
            self.count += 1;
        }

        self.map[target][dir] ^= true;
        self.map[next][inv(dir)] ^= true;
    }
}

fn annealing(input: &Input, initial_solution: AnnealingState, duration: f64) -> AnnealingState {
    let mut solution = initial_solution;
    let mut best_solution = solution.clone();
    let mut current_score = solution.calc_score(input);
    let mut best_score = current_score;
    let init_score = current_score;

    let mut all_iter = 0;
    let mut valid_iter = 0;
    let mut accepted_count = 0;
    let mut update_count = 0;
    let mut rng = Xoshiro256::new(42);

    let duration_inv = 1.0 / duration;
    let since = std::time::Instant::now();

    let temp0 = 3e4;
    let temp1 = 3e2;
    let mut inv_temp = 1.0 / temp0;

    loop {
        all_iter += 1;
        if (all_iter & ((1 << 4) - 1)) == 0 {
            let time = (std::time::Instant::now() - since).as_secs_f64() * duration_inv;
            let temp = f64::powf(temp0, 1.0 - time) * f64::powf(temp1, time);
            inv_temp = 1.0 / temp;

            if time >= 1.0 {
                break;
            }
        }

        // 変形
        let target = Coordinate::new(rng.gen_usize(0, N), rng.gen_usize(0, N));
        let dir = rng.gen_usize(1, 3);

        if !solution.can_flip(target, dir) {
            continue;
        }

        solution.flip(target, dir);

        // スコア計算
        let new_score = solution.calc_score(input);
        let score_diff = new_score - current_score;

        if score_diff >= 0 || rng.gen_bool(f64::exp(score_diff as f64 * inv_temp)) {
            // 解の更新
            current_score = new_score;
            accepted_count += 1;

            if best_score.change_max(current_score) {
                best_solution = solution.clone();
                update_count += 1;
            }
        } else {
            solution.flip(target, dir);
        }

        valid_iter += 1;
    }

    eprintln!("===== annealing =====");
    eprintln!("init score : {}", init_score);
    eprintln!("score      : {}", best_score);
    eprintln!("all iter   : {}", all_iter);
    eprintln!("valid iter : {}", valid_iter);
    eprintln!("accepted   : {}", accepted_count);
    eprintln!("updated    : {}", update_count);
    eprintln!("");

    best_solution
}

mod judge {
    use std::{
        cmp::Reverse,
        collections::BinaryHeap,
        io::{stdout, Write},
    };

    use crate::{
        grid::{inv, Coordinate, Map2d, ADJACENTS},
        Action, ChangeMinMax, Input, State, N,
    };

    pub(crate) trait Judge {
        fn read_input(&mut self) -> Input;
        fn update_state(&mut self, state: &mut State);
        fn apply(&mut self, action: Action);
    }

    pub(crate) struct OnlineJudge;

    impl Judge for OnlineJudge {
        fn read_input(&mut self) -> Input {
            Input::read_input()
        }

        fn update_state(&mut self, state: &mut State) {
            let (money, collaborator) = get!(i64, i64);

            if money == -1 && collaborator == -1 {
                panic!();
            }

            state.money = money;
            state.collaborator = collaborator;
        }

        fn apply(&mut self, action: Action) {
            println!("{}", action);
            stdout().flush().unwrap();
        }
    }

    pub(crate) struct SelfJudge {
        input: Input,
        state: State,
        turn: usize,
    }

    impl SelfJudge {
        pub(crate) fn new() -> Self {
            let input = Input {
                n: 0,
                t: 0,
                people: vec![],
            };
            let state = State::init();
            Self {
                input,
                state,
                turn: 0,
            }
        }
    }

    impl Judge for SelfJudge {
        fn read_input(&mut self) -> Input {
            self.input = Input::read_input();
            self.input.clone()
        }

        fn update_state(&mut self, state: &mut State) {
            self.turn += 1;
            eprintln!("[TURN {:3}]", self.turn);

            let mut distances = Map2d::new(vec![Map2d::new(vec![], N); N * N], N);

            for row0 in 0..N {
                for col0 in 0..N {
                    let mut dists = Map2d::new(vec![std::i32::MAX / 2; N * N], N);
                    let mut queue = BinaryHeap::new();
                    dists[Coordinate::new(row0, col0)] = 0;
                    queue.push(Reverse((0, Coordinate::new(row0, col0))));

                    while let Some(Reverse((dist, c))) = queue.pop() {
                        if dist > dists[c] {
                            continue;
                        }

                        for dir in 0..4 {
                            let next = c + ADJACENTS[dir];
                            let next_cost = dist + if self.state.map[c][dir] { 223 } else { 1000 };

                            if next.in_map(N) && dists[next].change_min(next_cost) {
                                queue.push(Reverse((next_cost, next)));
                            }
                        }
                    }

                    distances[Coordinate::new(row0, col0)] = dists;
                }
            }

            for &person in &self.input.people {
                let dist = distances[person.home][person.company];

                for count in 0.. {
                    let remain = dist - 223 * count;
                    if remain % 1000 == 0 {
                        self.state.money += 60 * count as i64;
                        break;
                    }
                }
            }

            eprintln!("money: {}", self.state.money);

            state.money = self.state.money;
            state.collaborator = self.state.collaborator;
        }

        fn apply(&mut self, action: Action) {
            eprintln!("action: {}", action);

            match action {
                Action::Construct(p, q) => {
                    let cost = self.state.calc_construction_cost();
                    if self.state.money < cost {
                        panic!();
                    }

                    self.state.money -= cost;

                    for dir in 0..4 {
                        if p + ADJACENTS[dir] == q {
                            self.state.map[p][dir] = true;
                            self.state.map[q][inv(dir)] = true;
                            return;
                        }
                    }
                }
                Action::Collaboration => {
                    self.state.collaborator += 1;
                }
                Action::Money => {
                    self.state.money += 50000;
                }
            }
        }
    }
}

#[allow(dead_code)]
pub mod grid {
    use std::ops::{Div, DivAssign, Mul, MulAssign};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Coordinate {
        pub row: usize,
        pub col: usize,
    }

    impl Coordinate {
        pub const fn new(row: usize, col: usize) -> Self {
            Self { row, col }
        }

        pub fn in_map(&self, size: usize) -> bool {
            self.row < size && self.col < size
        }

        pub const fn to_index(&self, size: usize) -> usize {
            self.row * size + self.col
        }

        pub const fn dist(&self, other: &Self) -> usize {
            Self::dist_1d(self.row, other.row) + Self::dist_1d(self.col, other.col)
        }

        const fn dist_1d(x0: usize, x1: usize) -> usize {
            (x0 as i64 - x1 as i64).abs() as usize
        }
    }

    impl Mul<usize> for Coordinate {
        type Output = Coordinate;

        fn mul(self, rhs: usize) -> Self::Output {
            Self::new(self.row * rhs, self.col * rhs)
        }
    }

    impl MulAssign<usize> for Coordinate {
        fn mul_assign(&mut self, rhs: usize) {
            self.row *= rhs;
            self.col *= rhs;
        }
    }

    impl Div<usize> for Coordinate {
        type Output = Coordinate;

        fn div(self, rhs: usize) -> Self::Output {
            Self::new(self.row / rhs, self.col / rhs)
        }
    }

    impl DivAssign<usize> for Coordinate {
        fn div_assign(&mut self, rhs: usize) {
            self.row /= rhs;
            self.col /= rhs;
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CoordinateDiff {
        pub dr: usize,
        pub dc: usize,
    }

    impl CoordinateDiff {
        pub const fn new(dr: usize, dc: usize) -> Self {
            Self { dr, dc }
        }

        pub const fn invert(&self) -> Self {
            Self::new(0usize.wrapping_sub(self.dr), 0usize.wrapping_sub(self.dc))
        }
    }

    impl std::ops::Add<CoordinateDiff> for Coordinate {
        type Output = Coordinate;

        fn add(self, rhs: CoordinateDiff) -> Self::Output {
            Coordinate::new(self.row.wrapping_add(rhs.dr), self.col.wrapping_add(rhs.dc))
        }
    }

    pub const ADJACENTS: [CoordinateDiff; 4] = [
        CoordinateDiff::new(!0, 0),
        CoordinateDiff::new(0, 1),
        CoordinateDiff::new(1, 0),
        CoordinateDiff::new(0, !0),
    ];

    pub const DIRECTIONS: [char; 4] = ['U', 'R', 'D', 'L'];

    pub fn inv(dir: usize) -> usize {
        dir ^ 2
    }

    #[derive(Debug, Clone)]
    pub struct Map2d<T> {
        pub width: usize,
        pub height: usize,
        map: Vec<T>,
    }

    impl<T> Map2d<T> {
        pub fn new(map: Vec<T>, width: usize) -> Self {
            let height = map.len() / width;
            debug_assert!(width * height == map.len());
            Self { width, height, map }
        }
    }

    impl<T> std::ops::Index<Coordinate> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: Coordinate) -> &Self::Output {
            &self.map[coordinate.row * self.width + coordinate.col]
        }
    }

    impl<T> std::ops::IndexMut<Coordinate> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: Coordinate) -> &mut Self::Output {
            &mut self.map[coordinate.row * self.width + coordinate.col]
        }
    }

    impl<T> std::ops::Index<&Coordinate> for Map2d<T> {
        type Output = T;

        #[inline]
        fn index(&self, coordinate: &Coordinate) -> &Self::Output {
            &self.map[coordinate.row * self.width + coordinate.col]
        }
    }

    impl<T> std::ops::IndexMut<&Coordinate> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, coordinate: &Coordinate) -> &mut Self::Output {
            &mut self.map[coordinate.row * self.width + coordinate.col]
        }
    }

    impl<T> std::ops::Index<usize> for Map2d<T> {
        type Output = [T];

        #[inline]
        fn index(&self, row: usize) -> &Self::Output {
            let begin = row * self.width;
            let end = begin + self.width;
            &self.map[begin..end]
        }
    }

    impl<T> std::ops::IndexMut<usize> for Map2d<T> {
        #[inline]
        fn index_mut(&mut self, row: usize) -> &mut Self::Output {
            let begin = row * self.width;
            let end = begin + self.width;
            &mut self.map[begin..end]
        }
    }
}

mod rand {
    pub(crate) struct Xoshiro256 {
        s0: u64,
        s1: u64,
        s2: u64,
        s3: u64,
    }

    impl Xoshiro256 {
        pub(crate) fn new(mut seed: u64) -> Self {
            let s0 = split_mix_64(&mut seed);
            let s1 = split_mix_64(&mut seed);
            let s2 = split_mix_64(&mut seed);
            let s3 = split_mix_64(&mut seed);
            Self { s0, s1, s2, s3 }
        }

        fn next(&mut self) -> u64 {
            let result = (self.s1 * 5).rotate_left(7) * 9;
            let t = self.s1 << 17;

            self.s2 ^= self.s0;
            self.s3 ^= self.s1;
            self.s1 ^= self.s2;
            self.s0 ^= self.s3;
            self.s2 ^= t;
            self.s3 = self.s3.rotate_left(45);

            result
        }

        pub(crate) fn gen_usize(&mut self, lower: usize, upper: usize) -> usize {
            assert!(lower < upper);
            let count = upper - lower;
            (self.next() % count as u64) as usize + lower
        }

        pub(crate) fn gen_i32(&mut self, lower: i32, upper: i32) -> i32 {
            assert!(lower < upper);
            let count = upper - lower;
            (self.next() % count as u64) as i32 + lower
        }

        pub(crate) fn gen_f64(&mut self) -> f64 {
            const UPPER_MASK: u64 = 0x3ff0000000000000;
            const LOWER_MASK: u64 = 0xfffffffffffff;
            let result = UPPER_MASK | (self.next() & LOWER_MASK);
            let result: f64 = unsafe { std::mem::transmute(result) };
            result - 1.0
        }

        pub(crate) fn gen_bool(&mut self, prob: f64) -> bool {
            self.gen_f64() < prob
        }
    }

    fn split_mix_64(x: &mut u64) -> u64 {
        *x += 0x9e3779b97f4a7c15;
        let mut z = *x;
        z = (z ^ z >> 30) * 0xbf58476d1ce4e5b9;
        z = (z ^ z >> 27) * 0x94d049bb133111eb;
        return z ^ z >> 31;
    }
}
