/*
Custom "threadsafe" union find structure.

- I want to do this with my custom IntN types to save space (so atomic operations are out)
- Ultimately I want this to be threadsafe, but let's not do that immediately 
  (maybe we'll have to lock to do atomic swapping?)
- Let's build this directly for my purposes and then make it generic later
- Operations I want to support:
	- Add Node 
	- Find root 
	- Union two nodes	
*/

use std::path::PathBuf;
use std::collections::HashMap;

pub(crate) struct UnionFind {
    pub parent: HashMap<(usize, usize), (usize, usize)>,
    pub rank: HashMap<(usize, usize), usize>,
}

impl UnionFind {
    pub fn new() -> Self {
        UnionFind {
            parent: HashMap::new(),
            rank: HashMap::new(),
        }
    }

    fn insert(&mut self, x: (usize, usize)) {
        if !self.parent.contains_key(&x) {
            self.parent.insert(x, x);
            self.rank.insert(x, 0);
        }
    }

    pub fn find(&mut self, x: (usize, usize)) -> (usize, usize) {
        if let Some(&parent) = self.parent.get(&x) {
            if parent != x {
                let root = self.find(parent);
                self.parent.insert(x, root);
                root
            } else {
                x
            }
        } else {
            self.insert(x);
            x
        }
    }

    pub fn union(&mut self, x: (usize, usize), y: (usize, usize)) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x != root_y {
            let rank_x = self.rank[&root_x];
            let rank_y = self.rank[&root_y];

            if rank_x < rank_y {
                self.parent.insert(root_x, root_y);
            } else if rank_x > rank_y {
                self.parent.insert(root_y, root_x);
            } else {
                self.parent.insert(root_y, root_x);
                *self.rank.get_mut(&root_x).unwrap() += 1;
            }
        }
    }

    pub fn get_ccs(&self) -> HashMap<(usize, usize), (usize, usize)> {
        let ccs : HashMap<(usize, usize), (usize, usize)> = HashMap::new();
        ccs
    }

}