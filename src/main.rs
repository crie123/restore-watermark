mod tests;

use ttf_parser::Face;
use std::fs;
use std::collections::HashMap;
use std::path::Path;
use rand::Rng;
use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;

// ============================================
// N-GRAM MODEL
// ============================================

#[derive(Default, Clone)]
pub struct NGramModel {
    pub n: usize,
    pub counts: HashMap<String, usize>,
    pub total: usize,
}

pub fn train_ngram(text: &str, n: usize) -> NGramModel {
    let mut model = NGramModel {
        n,
        counts: HashMap::new(),
        total: 0,
    };

    let chars: Vec<char> = text.chars().collect();

    for i in 0..chars.len().saturating_sub(n - 1) {
        let gram: String = chars[i..i + n].iter().collect();
        *model.counts.entry(gram).or_insert(0) += 1;
        model.total += 1;
    }

    model
}

pub fn ngram_score(text: &str, model: &NGramModel) -> f32 {
    let chars: Vec<char> = text.chars().collect();
    let mut score = 0.0;

    for i in 0..chars.len().saturating_sub(model.n - 1) {
        let gram: String = chars[i..i + model.n].iter().collect();
        let count = model.counts.get(&gram).copied().unwrap_or(1);
        score += (count as f32).ln();
    }

    score
}

// ============================================
// WATERMARK SIGNATURES
// ============================================

#[derive(Clone)]
pub struct AxisWatermark {
    pub lattice: Vec<f64>,
    pub strength: f64,
}

#[derive(Clone)]
pub struct MultiWatermark {
    pub axes: Vec<AxisWatermark>,
}

pub fn generate_multi_watermark(
    len: usize,
    seeds: &[u64],
    strength: f64,
) -> MultiWatermark {
    let axes = seeds
        .iter()
        .map(|&seed| {
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            let lattice = (0..len)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            AxisWatermark { lattice, strength }
        })
        .collect();

    MultiWatermark { axes }
}

pub fn apply_multi_watermark(
    signal: &mut [f64],
    wm: &MultiWatermark,
) {
    for axis in &wm.axes {
        for (v, w) in signal.iter_mut().zip(axis.lattice.iter()) {
            *v += w * axis.strength;
        }
    }
}

pub fn verify_multi_watermark(
    signal: &[f64],
    wm: &MultiWatermark,
) -> f64 {
    wm.axes.iter().map(|axis| {
        let mut corr = 0.0;
        let mut norm = 0.0;

        for (v, w) in signal.iter().zip(axis.lattice.iter()) {
            corr += v * w;
            norm += w * w;
        }

        corr / norm.sqrt()
    }).sum::<f64>() / wm.axes.len() as f64
}

pub fn normalize_signal(signal: &mut [f64]) {
    let norm = signal.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 0.0 {
        for v in signal {
            *v /= norm;
        }
    }
}

pub fn verify_with_mask(
    signal: &[f64],
    wm: &MultiWatermark,
    mask: &[bool],
) -> f64 {
    wm.axes.iter().map(|axis| {
        let mut corr = 0.0;
        let mut norm = 0.0;

        for ((&v, &w), &m) in signal.iter()
            .zip(axis.lattice.iter())
            .zip(mask.iter())
        {
            if m {
                corr += v * w;
                norm += w * w;
            }
        }

        if norm > 0.0 { corr / norm.sqrt() } else { 0.0 }
    }).sum::<f64>() / wm.axes.len() as f64
}

// ============================================
// SIGNAL TRANSFORMATIONS AND ATTACKS
// ============================================

pub fn add_noise(signal: &mut [f64], amplitude: f64) {
    let mut rng = rand::thread_rng();
    for v in signal {
        *v += rng.gen_range(-amplitude..amplitude);
    }
}

pub fn scale_signal(signal: &mut [f64], factor: f64) {
    for v in signal {
        *v *= factor;
    }
}

pub fn crop_signal(signal: &[f64], keep_ratio: f64) -> Vec<f64> {
    let keep = (signal.len() as f64 * keep_ratio) as usize;
    signal[..keep].to_vec()
}

pub fn permute_signal(signal: &mut [f64]) {
    let mut rng = rand::thread_rng();
    use rand::seq::SliceRandom;
    signal.shuffle(&mut rng);
}

pub fn recovery_ratio(
    original_score: f64,
    modified_score: f64,
) -> f64 {
    if original_score.abs() < 1e-6 {
        0.0
    } else {
        modified_score / original_score
    }
}

// ============================================
// PHASE-INVARIANT WATERMARK SCORING
// ============================================

pub fn phase_invariant_score(signal: &[f64], lattice: &[f64]) -> f64 {
    signal.iter()
        .zip(lattice)
        .map(|(s, v)| (s * v).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ============================================
// ANCHOR-AWARE WATERMARKING
// ============================================

#[derive(Clone, Debug)]
pub struct Anchor {
    pub text: String,
    pub bbox_width: f64,
    pub position: usize,
}

pub fn anchor_lattice(anchor: &Anchor, len: usize) -> Vec<f64> {
    let freq = anchor.bbox_width / 10.0;
    (0..len)
        .map(|i| ((i as f64) * freq).sin())
        .collect()
}

pub fn combined_anchor_lattice(
    anchors: &[Anchor],
    len: usize,
) -> Vec<f64> {
    let mut lattice = vec![0.0; len];
    for a in anchors {
        let local = anchor_lattice(a, len);
        for i in 0..len {
            lattice[i] += local[i];
        }
    }
    lattice
}

// ============================================
// PDF BBOX EXTRACTION
// ============================================

pub fn bbox_signal(widths: &[f64]) -> Vec<f64> {
    let norm = widths.iter().map(|w| w * w).sum::<f64>().sqrt();
    if norm > 0.0 {
        widths.iter().map(|w| w / norm).collect()
    } else {
        widths.to_vec()
    }
}

pub fn extract_bboxes_mock(widths: &[f64]) -> Vec<f64> {
    widths.to_vec()
}

// ============================================
// 3D MESH WATERMARKING
// ============================================

#[derive(Clone, Debug)]
pub struct Mesh {
    pub vertices: Vec<[f64; 3]>,
    pub edges: Vec<(usize, usize)>,
}

pub fn edge_lengths(mesh: &Mesh) -> Vec<f64> {
    mesh.edges.iter().map(|(a, b)| {
        let va = mesh.vertices[*a];
        let vb = mesh.vertices[*b];
        ((va[0] - vb[0]).powi(2)
            + (va[1] - vb[1]).powi(2)
            + (va[2] - vb[2]).powi(2)).sqrt()
    }).collect()
}

pub fn mesh_watermark(signal: &[f64], lattice: &[f64]) -> f64 {
    phase_invariant_score(signal, lattice)
}

// ============================================
// FFT-BASED BLOCK PROCESSING SYSTEM
// ============================================

pub fn split_into_blocks(signal: &[f64], block_size: usize) -> Vec<&[f64]> {
    signal
        .chunks(block_size)
        .filter(|b| b.len() == block_size)
        .collect()
}

pub fn fft_magnitude(block: &[f64]) -> Vec<f64> {
    use rustfft::{FftPlanner, num_complex::Complex};
    
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(block.len());

    let mut buffer: Vec<Complex<f64>> =
        block.iter().map(|&x| Complex::new(x, 0.0)).collect();

    fft.process(&mut buffer);

    buffer.iter().map(|c| c.norm()).collect()
}

pub fn block_energy(magnitudes: &[f64]) -> f64 {
    magnitudes.iter().map(|v| v * v).sum::<f64>().sqrt()
}

// ============================================
// MULTI-BASIS WATERMARKING SYSTEM
// ============================================

#[derive(Clone, Debug)]
pub struct Basis {
    pub lattice: Vec<f64>,
    pub weight: f64,
}

pub fn project(signal: &[f64], lattice: &[f64]) -> Vec<f64> {
    signal.iter()
        .zip(lattice)
        .map(|(s, l)| s * l)
        .collect()
}

pub fn score_block_multi_basis(
    block: &[f64],
    bases: &[Basis],
) -> f64 {
    bases.iter().map(|b| {
        let projected = project(block, &b.lattice);
        let mag = fft_magnitude(&projected);
        b.weight * block_energy(&mag)
    }).sum()
}

pub fn invariant_signature_score(
    signal: &[f64],
    bases: &[Basis],
    block_size: usize,
) -> f64 {
    let blocks = split_into_blocks(signal, block_size);

    let scores: Vec<f64> = blocks.iter()
        .map(|b| score_block_multi_basis(b, bases))
        .collect();

    // Median is robust to outliers and attacks
    if scores.is_empty() {
        return 0.0;
    }
    
    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal));
    sorted[sorted.len() / 2]
}

// ============================================
// ANCHORS AND QUANTIZATION
// ============================================

pub fn quantize(w: f32) -> i32 {
    (w * 10.0).round() as i32 // 0.1 px precision
}

pub fn anchor_bonus(
    text: &str,
    width: f32,
    anchors: &HashMap<i32, String>,
) -> f32 {
    let key = quantize(width);
    if let Some(anchor) = anchors.get(&key) {
        if anchor == text {
            return 5.0; // srong bonus for anchor match
        }
    }
    0.0
}

// ============================================
// DOCUMENT STRUCTURES AND STABILIZATION
// ============================================

#[derive(Clone)]
pub struct Line {
    pub observed_width: f32,
    pub beams: Vec<Beam>,
}

pub struct Document {
    pub lines: Vec<Line>,
}

pub fn stabilize_document(doc: &mut Document) {
    let mut anchors = HashMap::new();

    // collect best anchors from each line
    for line in &doc.lines {
        if let Some(best) = line.beams.first() {
            anchors.insert(quantize(line.observed_width), best.text.clone());
        }
    }

    eprintln!(" Found {} anchors for multi-line matching", anchors.len());

    // rescore beams based on anchors
    for line in &mut doc.lines {
        for beam in &mut line.beams {
            beam.score += anchor_bonus(
                &beam.text,
                line.observed_width,
                &anchors,
            );
        }

        line.beams.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    }
}

// ============================================
// PDF STRUCTURES AND INFERENCE
// ============================================

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct BBox {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct PdfLine {
    pub bbox: BBox,
    pub width: f32,
}

pub fn create_pdf_lines(widths: &[f32]) -> Vec<PdfLine> {
    widths
        .iter()
        .enumerate()
        .map(|(i, &w)| PdfLine {
            bbox: BBox {
                x: 0.0,
                y: (i as f32) * 20.0,
                w,
                h: 18.0,
            },
            width: w,
        })
        .collect()
}

// ============================================
// FONT LOADING, GLYPH MEASUREMENT, AND BEAM SEARCH
// ============================================

pub fn load_font(path: &str) -> Face<'static> {
    eprintln!(" Loading font: {}", path);
    
    if Path::new(path).exists() {
        let data = fs::read(path).expect("font read failed");
        return Face::parse(Box::leak(data.into_boxed_slice()), 0)
            .expect("font parse failed");
    }
    
    let alternatives = vec![
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\ArialMT.ttf",
        "C:\\Windows\\Fonts\\calibrib.ttf",
    ];
    
    for alt_path in alternatives {
        if Path::new(alt_path).exists() {
            eprintln!(" Using system font: {}", alt_path);
            let data = fs::read(alt_path).expect("font read failed");
            return Face::parse(Box::leak(data.into_boxed_slice()), 0)
                .expect("font parse failed");
        }
    }
    
    panic!(" Font not found: {}", path);
}

pub fn measure_text_kerning(
    text: &str,
    face: &Face,
    _glyphs: &HashMap<char, f32>,
    px_size: f32,
) -> f32 {
    let units_per_em = face.units_per_em() as f32;
    let scale = px_size / units_per_em;

    let mut total = 0.0;

    for ch in text.chars() {
        if let Some(glyph_id) = face.glyph_index(ch) {
            if let Some(advance) = face.glyph_hor_advance(glyph_id) {
                total += advance as f32 * scale;
            }
        }
    }

    total
}

pub fn build_glyph_widths(face: &Face, px_size: f32) -> HashMap<char, f32> {
    let units_per_em = face.units_per_em() as f32;
    let scale = px_size / units_per_em;

    let mut map = HashMap::new();

    let ranges = [
        (' '..='~'),              // ASCII
        ('А'..='Я'),              // cyrillic uppercase
        ('а'..='я'),
        ('Ё'..='Ё'),
        ('ё'..='ё'),
    ];

    for range in ranges {
        for ch in range {
            if let Some(glyph_id) = face.glyph_index(ch) {
                if let Some(advance) = face.glyph_hor_advance(glyph_id) {
                    map.insert(ch, advance as f32 * scale);
                }
            }
        }
    }

    map
}

pub fn find_candidates(
    target_width: f32,
    glyphs: &HashMap<char, f32>,
    dictionary: &[&str],
    tolerance: f32,
) -> Vec<(String, f32)> {
    let mut out = vec![];

    for &word in dictionary {
        let w: f32 = word.chars()
            .map(|c| glyphs.get(&c).copied().unwrap_or(0.0))
            .sum();
        let delta = (w - target_width).abs();

        if delta <= tolerance {
            out.push((word.to_string(), delta));
        }
    }

    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    out
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct ScoreWeights {
    pub width: f32,
    pub word_len: f32,
    pub spaces: f32,
}

#[derive(Clone)]
pub struct Beam {
    pub text: String,
    pub width: f32,
    pub score: f32,
}

#[allow(dead_code)]
pub fn score_text(
    text: &str,
    measured_width: f32,
    target_width: f32,
    weights: &ScoreWeights,
) -> f32 {
    let width_error = (measured_width - target_width).abs();
    let len = text.chars().count() as f32;
    let spaces = text.matches(' ').count() as f32;

    -weights.width * width_error
        - weights.word_len * len
        + weights.spaces * spaces
}

#[allow(dead_code)]
pub fn beam_search(
    face: &Face,
    _glyphs: &HashMap<char, f32>,
    px_size: f32,
    target_width: f32,
    alphabet: &[char],
    weights: &ScoreWeights,
    beam_width: usize,
    max_len: usize,
) -> Vec<Beam> {
    let mut beams = vec![Beam {
        text: String::new(),
        width: 0.0,
        score: 0.0,
    }];

    for _ in 0..max_len {
        let mut next = Vec::new();

        for beam in &beams {
            for &ch in alphabet {
                let mut new_text = beam.text.clone();
                new_text.push(ch);

                let new_width = measure_text_kerning(
                    &new_text,
                    face,
                    _glyphs,
                    px_size,
                );

                if new_width > target_width + 20.0 {
                    continue;
                }

                let score = score_text(
                    &new_text,
                    new_width,
                    target_width,
                    weights,
                );

                next.push(Beam {
                    text: new_text,
                    width: new_width,
                    score,
                });
            }
        }

        next.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        beams = next.into_iter().take(beam_width).collect();
    }

    beams
}

fn main() {
    eprintln!("\n╔════════════════════════════════════════════════════════════════╗");
    eprintln!("║        RESTORE_WATERMARK: Text restore system       ║");
    eprintln!("╚════════════════════════════════════════════════════════════════╝\n");

    eprintln!(" Initializing...");
    let face = load_font("fonts/DejaVuSans.ttf");
    
    let glyphs = build_glyph_widths(&face, 16.0);
    eprintln!(" Glyps loaded: {} symbols\n", glyphs.len());

    // Run extended tests with advanced watermarks
    tests::run_all_tests_with_advanced_watermarks(&face, &glyphs);
}
