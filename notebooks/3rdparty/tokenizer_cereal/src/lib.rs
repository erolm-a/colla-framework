extern crate serde_cbor;
extern crate bincode;
extern crate tokenizers;

extern crate itertools;
extern crate pyo3;

extern crate anyhow;

extern crate paste;
extern crate simple_error;

use simple_error::SimpleError;

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

#[derive(Deserialize)]
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

/// Automatically cast any form of error into a type error.
macro_rules! cast_errors {
    ($func:ident ( $param1:ident : $t1:ty $(, $param:ident : $t:ty )*) -> $resType:ty) => {
        paste! {
            #[pyfunction]
            fn $func( py: Python, $param1 : $t1, $($param : $t,)* ) -> PyResult<$resType> {
                match [<$func _helper>] (py, $param1 $(, $param)*) {
                    Ok(result) => Ok(result),
                    Err(e) => Err(exceptions::PyTypeError::new_err(e.to_string()))
                }
            }
        }
    };
}

#[pymodule]
fn tokenizer_cereal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tokenize_from_cbor_list, m)?)?;
    m.add_function(wrap_pyfunction!(get_token_slice, m)?)?;

    Ok(())
}

cast_errors!(tokenize_from_cbor_list(cbor_path: &str, ouput_path: &str, offset_list: Vec<u64> ) -> () );
cast_errors!(get_token_slice(cbor_path: &str, idx: usize, block_size: usize, content_block_idx: usize) -> (Vec<u32>, Vec<u32>));

fn tokenize_from_cbor_list_helper(
    _py: Python,
    cbor_path: &str,
    output_path: &str,
    offset_list: Vec<u64>,
) -> anyhow::Result<()> {
    let mut offset_list = offset_list.clone();
    // try to open the given cbor file
    let mut cbor_file = File::open(cbor_path)?;

    let wordpiece = WordPiece::from_files("vocab.txt") // TODO: move this somewhere else
        .unk_token("[UNK]".into())
        .build()
        .map_err(|e| simple_error_lined!(e))?;

    let mut tokenizer = Tokenizer::new(Box::new(wordpiece));
    // Make lowercase, ignore the chinese character problem
    tokenizer.with_normalizer(Box::new(BertNormalizer::new(true, false, true, true)));
    tokenizer.with_pre_tokenizer(Box::new(BertPreTokenizer));

    let file_length = cbor_file.seek(SeekFrom::End(0))?;
    cbor_file.seek(SeekFrom::Start(0))?;

    offset_list.push(file_length);

    let mut page_outputs = vec![];

    for (offset_start, offset_end) in offset_list.into_iter().tuple_windows() {
        let mut slice: Vec<u8> = vec![0; (offset_end - offset_start) as usize];

        cbor_file.read_exact(&mut slice).unwrap();

        let current_slice: PageFormat = serde_cbor::from_slice(&slice)?;

        println!("Read message {}", current_slice.text);

        let encoding = tokenizer
            .encode(EncodeInput::Single(current_slice.text), false)
            .map_err(|e| simple_error_lined!(e))?;

        let encoding_toks_offsets = encoding.get_offsets();
        let encoding_ids = encoding.get_ids();

        // TODO: it is not guaranteed that the link offsets won't change after normalization

        let link_embedding = extract_link_mask(encoding_toks_offsets, &current_slice.link_mentions);

        let page_output = PageFormatOutput {
            id: current_slice.id,
            tokens: encoding_ids.to_vec(),
            link_embedding,
        };

        print!("Found tokens: ");

        for encoding in &page_output.tokens {
            print!("{} ", encoding);
        }

        println!();

        print!("Found links: ");

        for link in &page_output.link_embedding {
            print!("{} ", link)
        }
        println!();

        page_outputs.push(page_output);
    }

    return write_slices(output_path, page_outputs);
}

fn write_slices(output_file: &str, page_outputs: Vec<PageFormatOutput>) -> anyhow::Result<()> {
    let output_file_tocs = output_file.to_owned() + ".toc";
    let mut output_file_stream = File::create(output_file)?;
    let output_file_tocs_stream = File::create(output_file_tocs)?;

    let mut offsets: Vec<u64> = page_outputs
        .into_iter()
        .scan(0 as usize, |prev_offset, page_output| {

            //wrap_cereal_error!(cereal::write_slice(&mut buf, &[page_output]))
            //    .unwrap();
            
            let size = bincode::serialized_size(&page_output).unwrap() as usize;

            bincode::serialize_into(&output_file_stream, &page_output)
                .unwrap();

            let old_offset = *prev_offset;
            *prev_offset += size;

            Some(old_offset as u64)
        })
        .collect();

    // push EOF position - will make some calculations easier this way
    offsets.push(output_file_stream.seek(SeekFrom::End(0))?.to_owned());
    // output_file_tocs_stream.write_all(offsets.as_slice())?;
    bincode::serialize_into(output_file_tocs_stream, &offsets)?;
    Ok(())
}

/// Get a chosen slice batch from a tokenized slice file
fn get_token_slice_helper(
    _py: Python,
    slice_file: &str,
    idx: usize,
    block_idx: usize,
    context_block_size: usize,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    let mut input_file = File::open(slice_file)?;
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
