extern crate bincode;
extern crate tokenizers;

extern crate itertools;
extern crate pyo3;

extern crate anyhow;

extern crate paste;
extern crate simple_error;

extern crate indicatif;

use simple_error::SimpleError;

use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::SeekFrom;
use std::cmp::*;

use serde::{Deserialize, Serialize};
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::tokenizer::{EncodeInput, Tokenizer};

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use itertools::Itertools;
use paste::paste;

use indicatif::ProgressBar;

#[derive(FromPyObject)]
struct PageFormat {
    // a page identifier
    id: u32,
    // the actual text to tokenize
    text: String,
    // the links as byte spans.
    // The first element is the link key, the second and third element are the
    // span of the link (in byte offsets since the beginning of the token).
    link_mentions: Vec<(u32, u32, u32)>,
}

#[derive(Deserialize, Serialize)]
struct PageFormatOutput {
    // a page identifier
    id: u32,
    // the tokens of the page
    tokens: Vec<u32>,
    // the resulting attention mask
    link_embedding: Vec<u32>,
}

// TODO: understand how to make function attributes

macro_rules! simple_error_lined {
    ($e:expr) => {
        SimpleError::new(format!("{} at line {}", $e.to_string(), line!()))
    };
}

/// This macro is an absolute hack that arises from my very poor knowledge of macros and my laziness.
/// Automatically cast any form of error into a type error. Also add python-style documentation.
macro_rules! cast_errors {
    ($func:ident ( $param1:ident : $t1:ty $(, $param:ident : $t:ty )*) -> $resType:ty, $doc:literal) => {
        paste! {
            #[pyfunction]
            #[doc=$doc]
            fn $func( py: Python, $param1 : $t1, $($param : $t,)* ) -> PyResult<$resType> {
                match [<$func _helper>] (py, $param1 $(, $param)*) {
                    Ok(result) => Ok(result),
                    Err(e) => Err(exceptions::PyTypeError::new_err(format!("{} at line {}", e.to_string(), line!())))
                }
            }
        }
    };
}


/// This module provides some convenience functions for tokenizing and accessing tokenized
/// Wikipedia pages, along with correct output spans.
#[pymodule]
fn tokenizer_cereal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize_from_iterator, m)?)?;
    m.add_function(wrap_pyfunction!(get_token_slice, m)?)?;
    m.add_function(wrap_pyfunction!(count_frequency, m)?)?;

    Ok(())
}

cast_errors!(tokenize_from_iterator(generator: &PyAny, output_path: &str, estimated_len: u64) -> Vec<u32>,
"Tokenize from a generator.\n
:param dictionary_generator a Python generator that generates a PageFormat (see lib.rs for details)\n
:param output_path the serialized output path\n
:param estimated_len if provided as -1 do not show a progress bar. Else the
       progress bar will assume that there are estimated_len elements to
       process.
:retuns a vector with all the article lengths.");


cast_errors!(get_token_slice(cereal_path: &str, idx: usize, block_size: usize,
                                    content_block_idx: usize) -> (Vec<u32>, Vec<u32>),
"Get a choen slice batch from a tokenized slice file.\n

:param cereal_path the path of the serialized tokens file.\n
:param idx the index of the article to use according to the previously generated\n
       TOC (whose path is cereal_path + \".toc\").\n
:param block_size the size of the blocks\n
:param content_block_idx the block idx. The resulting block may have a smaller\n
       size than the prescribed block size (true for last blocks).\n
:returns a pair of vectors: text tokens and link link target output.");

cast_errors!(count_frequency(cereal_path: &str) -> HashMap<u32, u32>,
"Count the frequency of a tokenized slice file.\n
\n
:param cereal_path the path of the serialized tokens file.\n
:return a frequency count dictionary. The keys are link ids and the values are the frequency count.");


fn tokenize_from_iterator_helper(
    _py: Python,
    iterator: &PyAny,
    output_path: &str,
    estimated_len: u64
) -> anyhow::Result<Vec<u32>> {

    let wordpiece = WordPiece::from_files("vocab.txt") // TODO: move this somewhere else
        .unk_token("[UNK]".into())
        .build()
        .map_err(|e| simple_error_lined!(e))?;

    let mut tokenizer = Tokenizer::new(Box::new(wordpiece));
    // Make lowercase, ignore the chinese character problem
    tokenizer.with_normalizer(Box::new(BertNormalizer::new(true, false, true, true)));
    tokenizer.with_pre_tokenizer(Box::new(BertPreTokenizer));


    // the number of articles to write at a time
    const BUFFER_SIZE : usize = 100;

    // TODO: what happens when passing -1?
    let pb = ProgressBar::new(estimated_len);

    let mut output_file_stream = File::create(output_path)?;
    let output_file_tocs_stream = File::create(output_path.to_owned() + ".toc")?;

    let mut lengths = vec![];
    let mut offsets = vec![0];

    for chunk in &iterator.iter()?.take(estimated_len as usize)
                                  .map(|i| i.and_then(PyAny::extract::<PageFormat>))
                                  .chunks(BUFFER_SIZE) {
        
        let page_inputs: Vec<PageFormat> = chunk.map(|page_output| page_output.unwrap()).collect();

        let inputs = page_inputs.as_slice().iter().map(|x| EncodeInput::Single(x.text.to_owned())).collect();

        let encodings = tokenizer.encode_batch(inputs, false)
                                 .map_err(|e| simple_error_lined!(e))?;

        let page_outputs : Vec<PageFormatOutput> = encodings.iter().zip(page_inputs.iter()).map(|(encoding, page_input)| {
            let encoding_toks_offsets = encoding.get_offsets();
            let encoding_ids = encoding.get_ids();

            let link_embedding = extract_link_mask(encoding_toks_offsets, &page_input.link_mentions);

            pb.inc(1);

            PageFormatOutput {
                id: page_input.id,
                tokens: encoding_ids.to_vec(),
                link_embedding,
            }
        }).collect();
        

        write_slices(&mut output_file_stream, &page_outputs,
                     &mut lengths, &mut offsets)?;
    }

    bincode::serialize_into(output_file_tocs_stream, &offsets)?;
 
    Ok(lengths)
}

// fn write_slices(output_file: &str, page_outputs: &Vec<PageFormatOutput>, block_size: u32) -> anyhow::Result<Vec<u32>> {
fn write_slices<T: Write + Seek>(
    output_file_stream: &mut T, page_outputs: &Vec<PageFormatOutput>,
    lenghts: &mut Vec<u32>, offsets: &mut Vec<usize>)-> anyhow::Result<()> {
    let pb = ProgressBar::new(page_outputs.len() as u64);

    offsets.extend(page_outputs.iter().scan(
        *offsets.last().unwrap(), |offset, page_output| {
            let size = bincode::serialized_size(&page_output).unwrap() as usize;

            // serialize_into gives endianness issues apparently
            let buf = bincode::serialize(&page_output).unwrap();
            output_file_stream.write(buf.as_slice()).unwrap();

            *offset += size;

            pb.inc(1);

            Some(*offset)
        }
    ));

    let blocks: Vec<u32> = page_outputs
        .into_iter()
        .map(|page_output| page_output.tokens.len() as u32)
        .collect();

    // push EOF position - will make some calculations easier this way
    // offsets.push(output_file_stream.seek(SeekFrom::End(0))?.to_owned());
    // output_file_tocs_stream.write_all(offsets.as_slice())?;
    // bincode::serialize_into(output_file_tocs_stream, &offsets)?;
    lenghts.extend(blocks);
    Ok(())
}

fn get_token_slice_helper(
    _py: Python,
    slice_file: &str,
    idx: usize,
    block_idx: usize,
    context_block_size: usize,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    let mut input_file = File::open(slice_file)?;
    // FIXME: we are opening and deserializing this file every time! Can we avoid this?
    let toc_file = File::open(slice_file.to_owned() + ".toc")?;
    let slices: Vec<u64> = bincode::deserialize_from(toc_file)?;

    input_file.seek(SeekFrom::Start(slices[idx]))?;

    let mut buf: Vec<u8> = vec![0_u8; (slices[idx+1] - slices[idx]) as usize];
    input_file.read_exact(&mut buf)?;
    let page_format : PageFormatOutput = bincode::deserialize(&buf)?;

    let start_idx = context_block_size * block_idx;
    let end_idx = min(start_idx + context_block_size, page_format.tokens.len());

    Ok((page_format.tokens[start_idx..end_idx].to_vec(),
        page_format.link_embedding[start_idx..end_idx].to_vec()))
}

/// Return a link output list
fn extract_link_mask(
    encoding_toks_offsets: &[(usize, usize)],
    link_positions: &Vec<(u32, u32, u32)>,
) -> Vec<u32> {
    let n = encoding_toks_offsets.len();
    let mut link_ids = vec![0; n];

    // Determine if the current link mention overlaps with the given tokens
    let mut current_link_pos_size = 0;

    'outer: for (idx, offset) in encoding_toks_offsets.iter().enumerate() {
        let (mut l, mut r);
        let mut link_position;

        'inner: loop {
            if current_link_pos_size >= link_positions.len() {
                break 'outer;
            }

            link_position = link_positions[current_link_pos_size];

            l = link_position.1 as usize;
            r = link_position.2 as usize;

            if r >= offset.0 {
                break 'inner;
            }

            current_link_pos_size += 1;
        }

        // If there is an overlap between the mention span and the token span
        if !(r < offset.0 || l > offset.1) {
            link_ids[idx] = link_position.0;
        }
    }

    return link_ids;
}

fn count_frequency_helper(
    _py: Python,
    cereal_path: &str
) -> anyhow::Result<HashMap<u32, u32>> {
    let mut slice = File::open(cereal_path)?;
    let toc_file = File::open(cereal_path.to_owned() + ".toc")?;
    let slice_offsets: Vec<usize> = bincode::deserialize_from(toc_file)?;

    let mut book_reviews = HashMap::new();

    for (offset_start, offset_end) in slice_offsets.into_iter().tuple_windows() {
        let mut buf: Vec<u8> = vec![0; offset_end - offset_start];
        slice.read_exact(buf.as_mut_slice())?;

        let parsed: PageFormatOutput = bincode::deserialize(&buf)?;
        for x in parsed.link_embedding {
            let counter = book_reviews.entry(x).or_insert(0);
            *counter += 1;
        }
    }

    Ok(book_reviews)
}
