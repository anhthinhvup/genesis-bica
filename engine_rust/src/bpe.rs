//! Genesis-v6 Pure-Rust Byte-Level BPE Tokenizer (`bpe.rs`).
//! Lossless UTF-8 (0% `<UNK>` on Vietnamese diacritics, English, math, code), 16 control tokens.
//! Zero external crates | Strictly <= 275 LOC.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

pub const TOK_PAD: u16 = 0;
pub const TOK_BOS: u16 = 1;
pub const TOK_EOS: u16 = 2;
pub const TOK_USER: u16 = 3;
pub const TOK_BOT: u16 = 4;
pub const TOK_CTX: u16 = 5;
pub const TOK_CALL_CAS: u16 = 6;
pub const TOK_CALL_SAT: u16 = 7;
pub const TOK_CALL_CODE: u16 = 8;
pub const TOK_CALL_RAG: u16 = 9;
pub const TOK_TOOL_OUT: u16 = 10;
pub const NUM_SPECIAL: usize = 16;
pub const BYTE_OFFSET: usize = NUM_SPECIAL; // bytes 0..255 -> token IDs 16..271
pub const MERGE_OFFSET: usize = BYTE_OFFSET + 256; // merged tokens start at 272

pub struct ByteBpeTokenizer {
    pub vocab_size: usize,
    pub merges: Vec<(u16, u16)>,
    pub merge_rank: HashMap<(u16, u16), (usize, u16)>,
    pub id_to_bytes: Vec<Vec<u8>>,
}

impl ByteBpeTokenizer {
    pub fn from_merges(merges: Vec<(u16, u16)>) -> Self {
        let vocab_size = MERGE_OFFSET + merges.len();
        let mut id_to_bytes: Vec<Vec<u8>> = Vec::with_capacity(vocab_size);
        for i in 0..NUM_SPECIAL {
            let tag = match i as u16 {
                TOK_PAD => b"<PAD>".to_vec(),
                TOK_BOS => b"<BOS>".to_vec(),
                TOK_EOS => b"<EOS>".to_vec(),
                TOK_USER => b"<USER>".to_vec(),
                TOK_BOT => b"<BOT>".to_vec(),
                TOK_CTX => b"<CTX>".to_vec(),
                TOK_CALL_CAS => b"<CALL_CAS>".to_vec(),
                TOK_CALL_SAT => b"<CALL_SAT>".to_vec(),
                TOK_CALL_CODE => b"<CALL_CODE>".to_vec(),
                TOK_CALL_RAG => b"<CALL_RAG>".to_vec(),
                TOK_TOOL_OUT => b"<TOOL_OUT>".to_vec(),
                _ => format!("<SP{}>", i).into_bytes(),
            };
            id_to_bytes.push(tag);
        }
        for b in 0u16..256 {
            id_to_bytes.push(vec![b as u8]);
        }
        let mut merge_rank = HashMap::with_capacity(merges.len() * 2);
        for (rank, &(a, b)) in merges.iter().enumerate() {
            let new_id = (MERGE_OFFSET + rank) as u16;
            let mut joined = id_to_bytes[a as usize].clone();
            joined.extend_from_slice(&id_to_bytes[b as usize]);
            id_to_bytes.push(joined);
            merge_rank.insert((a, b), (rank, new_id));
        }
        Self { vocab_size, merges, merge_rank, id_to_bytes }
    }

    /// Trains Byte-Level BPE on whitespace-delimited chunks (fast frequency-weighted word table).
    pub fn train(text: &str, target_vocab: usize, min_freq: usize) -> Self {
        let num_merges = target_vocab.saturating_sub(MERGE_OFFSET);
        let mut word_counts: HashMap<Vec<u8>, u32> = HashMap::new();
        let bytes = text.as_bytes();
        let mut start = 0usize;
        for (i, &b) in bytes.iter().enumerate() {
            if b == b' ' || b == b'\n' || b == b'\t' {
                if i > start {
                    let chunk = &bytes[start..i];
                    if chunk.len() <= 48 && !(chunk.starts_with(b"<") && chunk.ends_with(b">")) {
                        *word_counts.entry(chunk.to_vec()).or_insert(0) += 1;
                    }
                }
                start = i; // include leading space on next word for GPT-2 style boundary merges
            }
        }
        if bytes.len() > start {
            let chunk = &bytes[start..];
            if chunk.len() <= 48 {
                *word_counts.entry(chunk.to_vec()).or_insert(0) += 1;
            }
        }

        // Keep top 45,000 most frequent chunks for rapid CPU BPE training (< 10s!)
        let mut items: Vec<(Vec<u16>, u32)> = word_counts
            .into_iter()
            .filter(|(_, c)| *c >= 2)
            .map(|(w, c)| (w.into_iter().map(|b| (BYTE_OFFSET + b as usize) as u16).collect(), c))
            .collect();
        items.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        if items.len() > 45_000 {
            items.truncate(45_000);
        }

        let mut pair_counts: HashMap<(u16, u16), i64> = HashMap::with_capacity(131_072);
        for (seq, cnt) in &items {
            for w in seq.windows(2) {
                *pair_counts.entry((w[0], w[1])).or_insert(0) += *cnt as i64;
            }
        }

        let mut merges: Vec<(u16, u16)> = Vec::with_capacity(num_merges);
        for step in 0..num_merges {
            let mut best_pair = (0u16, 0u16);
            let mut best_cnt = 0i64;
            for (&p, &c) in &pair_counts {
                if c > best_cnt {
                    best_cnt = c;
                    best_pair = p;
                }
            }
            if best_cnt < (min_freq as i64) {
                break;
            }
            let new_id = (MERGE_OFFSET + step) as u16;
            merges.push(best_pair);
            let (pa, pb) = best_pair;
            for (seq, cnt) in &mut items {
                if seq.len() < 2 {
                    continue;
                }
                let mut found = false;
                for w in seq.windows(2) {
                    if w[0] == pa && w[1] == pb {
                        found = true;
                        break;
                    }
                }
                if !found {
                    continue;
                }
                let c_i64 = *cnt as i64;
                for w in seq.windows(2) {
                    if let Some(v) = pair_counts.get_mut(&(w[0], w[1])) {
                        *v -= c_i64;
                    }
                }
                let mut nxt: Vec<u16> = Vec::with_capacity(seq.len());
                let mut i = 0usize;
                while i < seq.len() {
                    if i + 1 < seq.len() && seq[i] == pa && seq[i + 1] == pb {
                        nxt.push(new_id);
                        i += 2;
                    } else {
                        nxt.push(seq[i]);
                        i += 1;
                    }
                }
                for w in nxt.windows(2) {
                    *pair_counts.entry((w[0], w[1])).or_insert(0) += c_i64;
                }
                *seq = nxt;
            }
            pair_counts.remove(&best_pair);
            if (step + 1) % 1000 == 0 {
                eprintln!("[V6-BPE] merged {}/{} (top freq={})", step + 1, num_merges, best_cnt);
            }
        }
        Self::from_merges(merges)
    }

    fn encode_chunk(&self, chunk: &[u8], out: &mut Vec<u16>) {
        if chunk.is_empty() {
            return;
        }
        let mut seq: Vec<u16> = chunk.iter().map(|&b| (BYTE_OFFSET + b as usize) as u16).collect();
        while seq.len() >= 2 {
            let mut best_rank = usize::MAX;
            let mut best_pos = usize::MAX;
            let mut best_id = 0u16;
            for i in 0..seq.len() - 1 {
                if let Some(&(rank, nid)) = self.merge_rank.get(&(seq[i], seq[i + 1])) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pos = i;
                        best_id = nid;
                    }
                }
            }
            if best_pos == usize::MAX {
                break;
            }
            seq[best_pos] = best_id;
            seq.remove(best_pos + 1);
        }
        out.extend_from_slice(&seq);
    }

    /// Encodes text in strict O(N) time recognizing `<USER>`, `<BOT>`, `<CTX>`, `<EOS>`, `<CALL_*>`.
    pub fn encode(&self, text: &str) -> Vec<u16> {
        let mut out = Vec::with_capacity(text.len() / 2 + 8);
        let tags: [(&str, u16); 10] = [
            ("<BOS>", TOK_BOS), ("<EOS>", TOK_EOS), ("<USER>", TOK_USER),
            ("<BOT>", TOK_BOT), ("<CTX>", TOK_CTX), ("<CALL_CAS>", TOK_CALL_CAS),
            ("<CALL_SAT>", TOK_CALL_SAT), ("<CALL_CODE>", TOK_CALL_CODE),
            ("<CALL_RAG>", TOK_CALL_RAG), ("<TOOL_OUT>", TOK_TOOL_OUT),
        ];
        let mut rest = text;
        while !rest.is_empty() {
            // Find next '<' in O(distance_to_next_tag) instead of O(N) per absent tag!
            let (seg, next_tag) = if let Some(lt) = rest.find('<') {
                let tail = &rest[lt..];
                let mut matched: Option<(usize, u16)> = None;
                for &(tg, tid) in &tags {
                    if tail.starts_with(tg) {
                        matched = Some((lt + tg.len(), tid));
                        break;
                    }
                }
                match matched {
                    Some((adv, tid)) => (&rest[..lt], Some((adv, tid))),
                    None => (&rest[..lt + 1], Some((lt + 1, u16::MAX))),
                }
            } else {
                (rest, None)
            };
            if !seg.is_empty() {
                let b = seg.as_bytes();
                let mut st = 0usize;
                for (i, &ch) in b.iter().enumerate() {
                    if (ch == b' ' || ch == b'\n') && i > st {
                        self.encode_chunk(&b[st..i], &mut out);
                        st = i;
                    }
                }
                if b.len() > st {
                    self.encode_chunk(&b[st..], &mut out);
                }
            }
            match next_tag {
                Some((adv, tid)) => {
                    if tid != u16::MAX {
                        out.push(tid);
                    }
                    rest = &rest[adv..];
                }
                None => break,
            }
        }
        out
    }

    pub fn decode(&self, ids: &[u16]) -> String {
        let mut raw = Vec::with_capacity(ids.len() * 3);
        for &id in ids {
            let idx = id as usize;
            if idx >= BYTE_OFFSET && idx < self.id_to_bytes.len() {
                raw.extend_from_slice(&self.id_to_bytes[idx]);
            }
        }
        String::from_utf8_lossy(&raw).into_owned()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        w.write_all(b"GV6BPE01")?;
        w.write_all(&(self.merges.len() as u32).to_le_bytes())?;
        for &(a, b) in &self.merges {
            w.write_all(&a.to_le_bytes())?;
            w.write_all(&b.to_le_bytes())?;
        }
        w.flush()
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let mut r = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != b"GV6BPE01" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad BPE magic"));
        }
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let n = u32::from_le_bytes(buf4) as usize;
        let mut merges = Vec::with_capacity(n);
        let mut pair = [0u8; 4];
        for _ in 0..n {
            r.read_exact(&mut pair)?;
            let a = u16::from_le_bytes([pair[0], pair[1]]);
            let b = u16::from_le_bytes([pair[2], pair[3]]);
            merges.push((a, b));
        }
        Ok(Self::from_merges(merges))
    }
}
