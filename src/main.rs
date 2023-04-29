use std::{
    fmt::Display,
    io::{stdout, Write},
};

use grid::{inv, Coordinate, Map2d, ADJACENTS};

use crate::rand::Xoshiro256;

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

    fn update(&mut self) {
        let (money, collaborator) = get!(i64, i64);

        if money == -1 && collaborator == -1 {
            panic!();
        }

        self.money = money;
        self.collaborator = collaborator;
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
    let input = Input::read_input();
    let mut state = State::init();

    for turn in 0..input.t {
        state.update();
        let action = get_action(&input, &state, turn);
        action.apply(&mut state);
        println!("{}", action);
        stdout().flush().unwrap();
    }
}

fn get_action(input: &Input, state: &State, turn: usize) -> Action {
    if turn >= 350 {
        return Action::Money;
    }

    if !state.can_construct() {
        return Action::Collaboration;
    }

    let mut current = [Coordinate::new(7, 6); 4];

    for _ in 0..N {
        for dir in 0..4 {
            let last = current[dir];
            current[dir] = current[dir] + ADJACENTS[dir];

            if current[dir].in_map(N) && !state.map[current[dir]][inv(dir)] {
                return Action::Construct(current[dir], last);
            }
        }
    }

    Action::Collaboration
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
