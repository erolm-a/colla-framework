extern crate bincode;
extern crate tokenizers;

extern crate itertools;
extern crate pyo3;

extern crate anyhow;

extern crate paste;
extern crate simple_error;

extern crate indicatif;

extern crate crossbeam;

extern crate log;

extern crate memmap;

extern crate rayon;

use simple_error::SimpleError;

use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::io::SeekFrom;
use std::cmp::*;

use serde::{Deserialize, Serialize};
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::processors::bert::BertProcessing;
use tokenizers::tokenizer::{Model, PostProcessor, Encoding};
// use tokenizers::tokenizer::{EncodeInput};


use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use itertools::Itertools;
use paste::paste;

use indicatif::ProgressBar;

use memmap::{Mmap};

use rayon::prelude::*;

#[derive(FromPyObject)]
struct PageFormat {
    // a page identifier
    id: u32,

    title: String, // TODO: find some usage for this field.
    /// The actual pretokenized text we are going to use...
    pretokenized_text: Vec<(String, (usize, usize))>,
    // the links as byte spans.
    // The first element is the link key, the second and third element are the
    // span of the link (in byte offsets since the beginning of the token).
    link_mentions: Vec<(u32, u32, u32)>,
}

#[derive(Deserialize, Serialize, Clone)]
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

/// Automatically cast any form of error into a type error. Also add python-style documentation.
/// This macro is an absolute hack that arises from my very poor knowledge of macros and my laziness.
macro_rules! cast_errors {
    ($func:ident ( $param1:ident : $t1:ty $(, $param:ident : $t:ty )*) -> $resType:ty) => {
        paste! {
            fn $func( $param1 : $t1, $($param : $t,)* ) -> PyResult<$resType> {
                match [<$func _helper>] ($param1 $(, $param)*) {
                    Ok(result) => Ok(result),
                    Err(e) => Err(exceptions::PyTypeError::new_err(format!("{} at line {}", e.to_string(), line!())))
                }
            }
        }
    };

    (PYFUNC $func:ident ($param1:ident : $t1:ty $(, $param:ident : $t:ty )*) -> $resType:ty) => {
        paste! {
            #[pyfunction]
            fn $func($param1 : $t1, $($param : $t,)* ) -> PyResult<$resType> {
                match [<$func _helper>] ($param1 $(, $param)*) {
                    Ok(result) => Ok(result),
                    // Err(e) => Err(exceptions::PyTypeError::new_err(format!("{} at line {}", e.to_string(), line!())))
                    Err(e) => panic!("Happened error {} at line {} ", e.to_string(), line!())
                }
            }
        }
    };
}


#[pyclass]
struct TokenizerCereal {
    slice_file : std::fs::File,

    slice_file_mmap: Mmap,

    #[pyo3(get)]
    slice_offsets: Vec<usize>,

    #[pyo3(get)]
    article_lengths: Vec<u32>
}


#[pymethods]
impl TokenizerCereal {
    #[new]
    /// Create a new Tokenizer. 
    /// 
    /// :param slice_path the path of the outputs
    /// :param iterator the generator to use
    /// :param estimated_len the estimated number of article to preprocess. It only has cosmetic reasons for the progress bar.
    fn new(slice_path: &str, iterator: &PyAny, estimated_len: usize, py: Python) -> TokenizerCereal {

        let article_lenghts = tokenize_from_iterator(py, iterator, slice_path, estimated_len).unwrap();

        // serialize article lengths
        {
            let article_lenghts_file = File::create(slice_path.to_owned() + ".lenghts").unwrap();
            bincode::serialize_into(article_lenghts_file, &article_lenghts).unwrap();
        }

        // let slice_file = File::open(slice_path).unwrap();
        let toc_file = File::open(slice_path.to_owned() + ".toc").unwrap();

        let slice_offsets = bincode::deserialize_from(toc_file).unwrap();
        
        let slice_file = File::open(slice_path).unwrap();

        let mmap = unsafe {
            memmap::MmapOptions::new()
                                .map(&slice_file).unwrap()
        };

        TokenizerCereal {
            slice_file: slice_file,
            slice_file_mmap: mmap,
            slice_offsets: slice_offsets,
            article_lengths: article_lenghts
        }
    }

    /// Get a chosen slice batch from a tokenized slice file.
    /// This is the go-to method for implementing a map-based dataloader.
    /// :param idx the index of the article to use according to the previously generated
    ///        TOC (whose path is cereal_path + \".toc\").
    /// :param block_idx the block idx. The resulting block may have a smaller
    ///        size than the prescribed block size (true for last blocks).
    /// :param content_block_size the size of the blocks
    /// :returns a pair of vectors: text tokens and link link target output.
    fn get_slice(&mut self, idx: usize, block_idx: usize, content_block_size: usize)
                 -> PyResult<(Vec<u32>, Vec<u32>)> {
        get_token_slice(&mut self.slice_file_mmap, &self.slice_offsets, idx, block_idx, content_block_size)
    }

    /// Get the next slice batch from a tokenized slice file.
    /// :returns a pair of vectors: text tokens and link link target output.
    fn get_next_slice(&mut self)
                -> PyResult<(Vec<u32>, Vec<u32>)> {
        get_next_token_slice(&self.slice_file)
    }

    /// Count the frequency of a tokenized slice file.
    ///
    /// :returns a frequency count dictionary. The keys are link ids and the values are the
    ///          frequency count.
    fn get_frequency_count(&mut self) -> PyResult<HashMap<u32, u32>> {
        let file = &mut self.slice_file;
        file.seek(SeekFrom::Start(0))?;
        let output = count_frequency(file, &self.slice_offsets);
        output
    }
}

/// This module provides some convenience functions for tokenizing and accessing tokenized
/// Wikipedia pages, along with correct output spans.
#[pymodule]
fn tokenizer_cereal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TokenizerCereal>()?;
    m.add_function(wrap_pyfunction!(get_default_tokenizer, m)?)?;

    Ok(())
}

cast_errors!(tokenize_from_iterator(py: Python, generator: &PyAny, output_path: &str, estimated_len: usize) -> Vec<u32>);

cast_errors!(PYFUNC get_default_tokenizer(slice_path: &str) -> TokenizerCereal);


cast_errors!(get_token_slice(input_file: &Mmap, // required mutability for seeks
    slice_offsets: &Vec<usize>,
    idx: usize,
    block_idx: usize,
    context_block_size: usize) -> (Vec<u32>, Vec<u32>));

cast_errors!(get_next_token_slice(input_file: &File) -> (Vec<u32>, Vec<u32>));

cast_errors!(count_frequency(
    slice: &mut File,
    slice_offsets: &Vec<usize>) -> HashMap<u32, u32>);

fn tokenize_from_iterator_helper(
    py: Python,
    iterator: &PyAny,
    output_path: &str,
    estimated_len: usize
) -> anyhow::Result<Vec<u32>> {

    let wordpiece = WordPiece::from_files("vocab.txt") // TODO: move this somewhere else
        .unk_token("[UNK]".into())
        .build()
        .map_err(|e| simple_error_lined!(e))?;

    /*
    let mut tokenizer = Tokenizer::new(Box::new(wordpiece));
    // Make lowercase, ignore the chinese character problem
    tokenizer.with_normalizer(Box::new(BertNormalizer::new(true, false, true, true)));
    tokenizer.with_pre_tokenizer(Box::new(BertPreTokenizer));
    */
    let postprocessor = BertProcessing::new(
        ("[SEP]".to_owned(), 102),
        ("[CLS]".to_owned(), 101)
    );

    

    // the number of articles to write at a time
    const BUFFER_SIZE : usize = 500;

    let pb = ProgressBar::new(estimated_len as u64);

    let mut output_file_stream = File::create(output_path)?;
    let output_file_tocs_stream = File::create(output_path.to_owned() + ".toc")?;

    let mut lengths = vec![];
    let mut offsets = vec![0];

    for chunk in &iterator.iter()?.take(estimated_len)
                                  .map(|i| i.and_then(PyAny::extract::<PageFormat>))
                                  .chunks(BUFFER_SIZE) {
        
        // NOTE: Failing pages are silently ignored.
        // But at least do not silent the error!
        let page_inputs: Vec<PageFormat> = chunk.filter_map(|page_output| match page_output {
            Ok(value) => Some(value),
            Err(error) => {
                log::warn!("Got a Python error while tokenizing. This entry will be ignored. Read the stacktrace below for details.");
                error.print(py);
                None
            }
        }).collect();

        // let inputs = page_inputs.as_slice().iter().map(|x| EncodeInput::Single(x.text.to_owned())).collect();

        let tokenized_inputs: Vec<Vec<(String, (usize, usize))>> = page_inputs.iter().map(|x| x.pretokenized_text.clone()).collect();

        let encodings: Vec<Encoding> = tokenized_inputs.into_par_iter()
                       .filter_map(|page_input| wordpiece.tokenize(page_input).ok())
                       .map(|tokenized| Encoding::from_tokens(tokenized, 0))
                       .filter_map(|encoded| postprocessor.process(encoded, None, true).ok())
                       .collect();

        //let encodings = tokenizer.encode_batch(inputs, true)
        //                         .map_err(|e| simple_error_lined!(e))?;

        let page_outputs : Vec<PageFormatOutput> = encodings.iter()
            .zip(page_inputs.iter())
            .filter_map(|(encoding, page_input)| {
                let encoding_ids = encoding.get_ids();

                pb.inc(1);

                if encoding_ids.len() > 0 {
                    let encoding_toks_offsets = encoding.get_offsets();
                    let link_embedding = extract_link_mask(encoding_toks_offsets, &page_input.link_mentions);
                    Some(PageFormatOutput {
                        id: page_input.id,
                        tokens: encoding_ids.to_vec(),
                        link_embedding,
                    })
                } else {
                    log::warn!("page_input {} yielded an empty token list. Skipping", page_input.id);
                    None
                }
            }).collect();
        

        write_slices(&mut output_file_stream, &page_outputs,
                     &mut lengths, &mut offsets)?;
    }

    bincode::serialize_into(output_file_tocs_stream, &offsets)?;
 
    Ok(lengths)
}

fn get_default_tokenizer_helper(slice_path: &str) -> anyhow::Result<TokenizerCereal>
{

    let article_lenghts : Vec<u32>;
    let slice_offsets : Vec<usize>;

    {
        let article_lenghts_file = File::open(slice_path.to_owned() + ".lenghts")?;
        article_lenghts = bincode::deserialize_from(article_lenghts_file)?;
    }

    {
        let slice_toc_file = File::open(slice_path.to_owned() + ".toc")?;
        slice_offsets = bincode::deserialize_from(slice_toc_file)?;
    }


    let slice_file = File::open(slice_path).unwrap();

    let mmap = unsafe {
        memmap::MmapOptions::new()
                            .map(&slice_file).unwrap()
    };


    Ok(TokenizerCereal {
        slice_file: slice_file,
        slice_file_mmap: mmap,
        slice_offsets: slice_offsets,
        article_lengths: article_lenghts
    })
}



// fn write_slices(output_file: &str, page_outputs: &Vec<PageFormatOutput>, block_size: u32) -> anyhow::Result<Vec<u32>> {
fn write_slices<T: Write + Seek>(
    output_file_stream: &mut T,
    page_outputs: &Vec<PageFormatOutput>,
    lenghts: &mut Vec<u32>,
    offsets: &mut Vec<usize>)
        -> anyhow::Result<()> {
    let pb = ProgressBar::new(page_outputs.len() as u64);

    offsets.extend(page_outputs.iter().scan(
        // offsets always starts with a 0
        *offsets.last().unwrap(), |offset, page_output| {
            let size = bincode::serialized_size(&page_output).unwrap() as usize;

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
    // input_file: &mut File, // required mutability for seeks
    input_file: &Mmap,
    slice_offsets: &Vec<usize>,
    idx: usize,
    block_idx: usize,
    context_block_size: usize,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {


    /*
    input_file.seek(SeekFrom::Start(slice_offsets[idx] as u64))?;

    let mut buf: Vec<u8> = vec![0_u8; (slice_offsets[idx+1] - slice_offsets[idx]) as usize];

    input_file.read_exact(&mut buf)?;
    let page_format : PageFormatOutput = bincode::deserialize(&buf)?;
    */

    let page_format : PageFormatOutput = bincode::deserialize(&input_file[slice_offsets[idx]..slice_offsets[idx+1]])?;

    let start_idx = context_block_size * block_idx;
    let end_idx = min(start_idx + context_block_size, page_format.tokens.len());

    if start_idx >= end_idx {
        println!("HEY THERE! I am being called with params {} {}", idx, block_idx);
        println!("The fetched page_format at index {} has size {}", idx, page_format.tokens.len());
        println!("The calculated start_idx and end_idx are {} and {}", start_idx, end_idx);

        //println!("This ")
        //std::intrinsics::breakpoint();
    }

    Ok((page_format.tokens[start_idx..end_idx].to_vec(),
        page_format.link_embedding[start_idx..end_idx].to_vec()))
}

fn get_next_token_slice_helper(
    input_file: &File,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    let page_format : PageFormatOutput = bincode::deserialize_from(input_file)  ?;

    Ok((page_format.tokens, page_format.link_embedding))
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
    slice: &mut File,
    slice_offsets: &Vec<usize>
) -> anyhow::Result<HashMap<u32, u32>> {
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
