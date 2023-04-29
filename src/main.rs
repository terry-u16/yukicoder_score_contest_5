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

fn main() {}

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
