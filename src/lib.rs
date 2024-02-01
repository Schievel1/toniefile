//! # Toniefile
//!
//! The Toniefile crate provides methods to write audio data in a format that can be played by a
//! Toniebox. The Toniefile format is a modified Ogg Opus file format with a [Protobuf](https://protobuf.dev/) header.
//! However the [Ogg pages](https://en.wikipedia.org/wiki/Ogg_page) must be exactly 4096 bytes long and must begin
//! and end at a 4096 byte mark in the file.
//! (except for the first page, which starts at 0x1200 and ends at 0x1FFF)
//! The protobuf header itself must be padded to 4096 bytes as well and the Ogg header and Ogg comment page must
//! be padded to 200 bytes.
//! This results in a file like this:
//! ```text
//! Address   Data                                               ASCII
//! 00000000  00 00 0f fc 0a 14 04 ef  db a5 67 5d a9 a7 96 1e  |..........g]....|
//! ^ The protobuf header (starts at 0x04) until 0x0FFF.
//! * 0x00 - 0x03 are the length of the header in bytes
//! 00001000  4f 67 67 53 00 02 00 00  00 00 00 00 00 00 78 56  |OggS..........xV|
//! ^ Opus Header and Opus Comment until 0x11FF
//! 00001200  4f 67 67 53 00 00 40 38  00 00 00 00 00 00 78 56  |OggS..@8......xV|
//! ^ First payload audio data until 0x1FFF
//! 00002000  4f 67 67 53 00 00 40 38  00 00 00 00 00 00 78 56  |OggS..@8......xV|
//! ^ Second payload audio data until 0x2FFF
//! 00003000  4f 67 67 53 00 00 40 38  00 00 00 00 00 00 78 56  |OggS..@8......xV|
//! ^ And so on
//! ```
//!
//! # Usage
//! The Toniefile struct takes anything that implements the Write and Seek traits as a writer at it's creation.
//! When one of the methods encode() or finalize() is called, it writes to the writer.
//! During its creation it takes a unique audio id and an optional Vector of  user comments as arguments.
//! The audio id can be an arbitrary uint32 and is used to identify the audio file on the Toniebox.
//! (Use whatever you want here)
//! The user comment is a Vector auf &str that is written to the Ogg comment page, it can be empty.
//! ```text
//! fn fill_single_buffer_toniefile() {
//!     // create a file
//!     let file = File::create(~/my_little_toniefile").unwrap();
//!     // create a toniefile struct
//!     let mut toniefile = Toniefile::new (
//!         file,
//!         0x12345678,
//!         Some(vec!["my comment", "another comment"])
//!     )
//!     .unwrap();
//!
//!     // read samples, this can vary depending on your use case
//!     // for example use the wav crate to read a whole wav file into memory
//!     let mut f = File::open("~/my_little_samplesfile").expect("no file found");
//!     let (_, samples) = wav::read(&mut f).unwrap();
//!     samples.try_into_sixteen().unwrap()
//!     let samples: Vec<i16> = samples.to_str().unwrap();
//!
//!     // write samples to toniefile
//!     let res = toniefile.encode(&samples);
//!     // finalize the toniefile
//!     toniefile.finalize().unwrap();
//! }
//! ```
//! It is also possible to call encode() multiple times with smaller buffers. This can be useful if you
//! don't want to have the whole samples file in memory at once.
//! We can use new_simple() to create a Toniefile with a random audio id and no user comments if we don't
//! care about them.
//! Here we use the WavReader from the hound crate (<https://crates.io/crates/hound>) to read the samples
//! in chunks.
//! ```text
//! fn read_and_fill_chunks_toniefile() {
//!     let file = File::create("~/my_little_toniefile").unwrap();
//!     let mut toniefile = Toniefile::new_simple(file)
//!     .unwrap();
//!
//!     let mut f = File::open("~/my_little_samplesfile").unwrap();
//!     let mut wav_reader = hound::WavReader::new(&mut f).unwrap();
//!     let total_samples = wav_reader.duration();
//!     let mut wav_iter = wav_reader.samples::<i16>();
//!     let mut samples_read = 0;
//!     while samples_read < total_samples {
//!         let mut window = vec![];
//!         for _ in 0..TONIEFILE_FRAME_SIZE * OPUS_CHANNELS {
//!             window.push(wav_iter.next().unwrap().unwrap());
//!         }
//!         let res = toniefile.encode(&window);
//!         assert!(res.is_ok());
//!         samples_read += window.len() as u32;
//!     }
//!
//!     toniefile.finalize().unwrap();
//! }
//! ```
//! Lastly it is also possible to let the Toniefile struct write into a vector, instead of a file on disk as long as the vector
//! implements the Write and Seek traits. We can use the Cursor struct from the std library for this.
//! ```
//! use std::io::{Cursor, Seek, SeekFrom, Write};
//! use toniefile::{Toniefile, ToniefileError};
//!
//! fn fill_vector_toniefile() -> Result<(), ToniefileError> {
//!     let myvec: Vec<u8> = vec![];
//!     let cursor = Cursor::new(myvec);
//!     let mut toniefile = Toniefile::new(cursor, 0x12345678, None)?;
//!     let samples: Vec<i16> = vec![0; 48000 * 2 * 60]; // 60 seconds of finest silence
//!     let res = toniefile.encode(&samples);
//!     assert!(res.is_ok());
//!     toniefile.finalize_no_consume()?;
//!     // do something with the vector e.g. writing it to disk or sending it over the network
//!     Ok(())
//! }
//! ```
//!

use audiopus::packet::samples_per_frame;
use audiopus::repacketizer::packet_pad;
use audiopus::{coder::Encoder as OpusEnc, ffi, Bitrate, SampleRate};
use byteorder::{BigEndian, ByteOrder, LittleEndian};
use libogg::{Packet as OggPacket, Stream as OggStream};
use prost::Message;
use rand::Rng;
use sha1::digest::FixedOutputReset;
use sha1::{Digest, Sha1};
use std::ffi::CStr;
use std::fmt::{Display, Formatter};
use std::io::{Cursor, Read};
use std::{
    fs::File,
    io::{Seek, SeekFrom, Write},
};
use thiserror::Error;
use toniehead::TonieboxAudioFileHeader;

const OPUS_FRAME_SIZE_MS: usize = 60;
const OPUS_FRAME_SIZE_MS_OPUS: i32 = 5006; // This is a special value for 60ms see opus_defines.h
const OPUS_SAMPLE_RATE: usize = 48000;
const OPUS_BITRATE: u32 = 96000;
const OPUS_FRAME_SIZE: usize = OPUS_SAMPLE_RATE * OPUS_FRAME_SIZE_MS / 1000;
const OPUS_CHANNELS: usize = 2;
const OPUS_PACKET_PAD: usize = 64;
const OPUS_PACKET_MINSIZE: usize = 64;

const TONIEFILE_FRAME_SIZE: usize = 4096;
const TONIEFILE_MAX_CHAPTERS: usize = 100;
const TONIEFILE_PAD_END: usize = 64;
const CONTENT_LENGTH_MAX: i32 = i32::MAX;
const TONIE_LENGTH_MAX: i32 = CONTENT_LENGTH_MAX - 0x1000;

const COMMENT_LEN: usize = 0x1B4;

const SHA1_DIGEST_SIZE: usize = 20;

#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ToniefileError {
    #[error("Comment of length {0} and buffer of length {1} won't fit in {2}")]
    CommentWontFit(usize, usize, usize),
    #[error("Max number of chapters reached")]
    MaxChaptersReached,
    #[error("Not enough space in this block")]
    NotEnoughSpace,
    #[error("Encoded frame size mismatch! Got {0}, should be {1}")]
    FrameSizeDontMatch(usize, usize),
    #[error("Unexpected small padding at granule position{0} ( {1}sec)")]
    SmallPaddingError(u64, u64),
    #[error("Block alignment error at position {:#010x}", .0)]
    BlockAlignmentError(u64),
    #[error(transparent)]
    OpusEncoderError(#[from] audiopus::Error),
    #[error(transparent)]
    IOError(#[from] std::io::Error),
    #[error(transparent)]
    ProstEncodeError(#[from] prost::EncodeError),
    #[error(transparent)]
    ProstDecodeError(#[from] prost::DecodeError),
    #[error(transparent)]
    Utf8Error(#[from] std::str::Utf8Error),
}

pub mod toniehead {
    #![allow(non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/toniehead.rs"));
    use std::fmt::{Display, Formatter};
    impl Display for TonieboxAudioFileHeader {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "audio_id: {}\naudio length: {} kbyte\ntrack_page_nums: {:?}\nsha1_hash: {:?}\nfill length: {:?} }}",
                self.audio_id, self.num_bytes / 1024, self.track_page_nums, self.sha1_hash, self.fill.len()
            )
        }
    }
}

#[derive(Debug)]
pub struct Toniefile<F: Write + Seek> {
    writer: F,
    file_position: u64,
    audio_length: u32,
    // opus
    opus_encoder: OpusEnc,
    audio_frame: [i16; OPUS_FRAME_SIZE * OPUS_CHANNELS],
    audio_frame_used: usize,
    // ogg
    ogg_stream: OggStream,
    ogg_granulepos: u64,
    ogg_packet_count: i64,
    // header
    taf_header: TonieboxAudioFileHeader,
    sha1_ctx: Sha1,
    // current page number
    taf_page_number: u32,
}

impl<F: Write + Seek> Display for Toniefile<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "audio_id: {}\naudio_length: {} bytes\ntaf_page_number: {}\nfile_position: {}",
            self.taf_header.audio_id, self.audio_length, self.taf_page_number, self.file_position
        )
    }
}

impl Toniefile<File> {
    /// Associated function to parse the header of a Toniefile and return it
    ///
    /// # Errors
    ///
    /// Returns an error if the file is not seekable or a header could not be found.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::io::{Cursor, Seek, SeekFrom, Write};
    /// # use toniefile::{Toniefile, ToniefileError};
    /// #
    /// # fn parse_my_header() -> Result<(), ToniefileError> {
    /// #    let myvec: Vec<u8> = vec![];
    /// #    let cursor = Cursor::new(myvec);
    /// #    let mut toniefile = Toniefile::new(cursor, 0x12345678, None)?;
    /// #    let samples: Vec<i16> = vec![0; 48000 * 2 * 60]; // 60 seconds of finest silence
    /// #    let res = toniefile.encode(&samples);
    /// #    assert!(res.is_ok());
    /// #    toniefile.finalize_no_consume()?;
    ///     // you can also pass in a std::fs::File here
    ///     let header = Toniefile::parse_header(&mut toniefile.writer())?;
    ///     assert_eq!(header.audio_id, 0x12345678);
    /// #   Ok(())
    /// # }
    ///
    /// ```
    pub fn parse_header<R: Read + Seek>(
        reader: &mut R,
    ) -> Result<TonieboxAudioFileHeader, ToniefileError> {
        reader.rewind()?;
        let mut len_bf = [0u8; 4];
        reader.read_exact(&mut len_bf)?;
        let proto_size = BigEndian::read_u32(&len_bf) as usize;
        let mut buffer = vec![0u8; proto_size];
        reader.read_exact(&mut buffer)?;
        let header = TonieboxAudioFileHeader::decode(&mut Cursor::new(buffer))?;
        Ok(header)
    }

    /// Associated function to get the audio data out of a Toniefile.
    /// After a call to this function the passed is stream will be seeked to the end of it.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::io::{Cursor, Seek, SeekFrom, Write};
    /// # use toniefile::{Toniefile, ToniefileError};
    /// #
    /// # fn get_audio_data() -> Result<(), ToniefileError> {
    /// #    let myvec: Vec<u8> = vec![];
    /// #    let cursor = Cursor::new(myvec);
    /// #    let mut toniefile = Toniefile::new(cursor, 0x12345678, None)?;
    /// #    let samples: Vec<i16> = vec![0; 48000 * 2 * 60]; // 60 seconds of finest silence
    /// #    let res = toniefile.encode(&samples);
    /// #    assert!(res.is_ok());
    /// #    toniefile.finalize_no_consume()?;
    ///     // you can also pass in a std::fs::File here
    ///     let audio_data = Toniefile::extract_audio(&mut toniefile.writer())?;
    ///     // do something with the audio data e.g. write it to a file
    ///     // let mut audio_file = File::create("audio.ogg")?;
    ///     // audio_file.write_all(&audio_data)?;
    ///     // audio_file.close()?;
    ///     assert!(audio_data.starts_with(b"OggS")); // Ogg files start with the OggS magic
    /// #   Ok(())
    /// # }
    ///
    /// ```
    pub fn extract_audio<R: Read + Seek>(reader: &mut R) -> Result<Vec<u8>, ToniefileError> {
        const LEN_LENGTH: usize = 4;
        reader.rewind()?;
        let mut len_bf = [0u8; 4];
        reader.read_exact(&mut len_bf)?;
        let audio_start = BigEndian::read_u32(&len_bf) as usize + LEN_LENGTH;

        let mut audio_data = vec![];
        reader.seek(SeekFrom::Start(audio_start as u64))?;
        reader.read_to_end(&mut audio_data)?;
        Ok(audio_data)
    }
}

impl<F: Write + Seek> Toniefile<F> {
    /// Create a new Toniefile with a random number as audio id and no user comments
    ///
    /// # Errors
    ///
    /// Returns an error if the space for comments in the file would overflow. (382 bytes, but
    /// every comment needs 4 bytes extra for the length indicator)
    ///
    /// # Example
    ///
    /// ```
    /// # use std::io::Cursor;
    /// # use toniefile::Toniefile;
    /// #
    /// # fn create_simple_toniefile() {
    ///     let myvec: Vec<u8> = vec![];
    ///     let cursor = Cursor::new(myvec);
    ///     let toniefile = Toniefile::new_simple(cursor);
    ///     assert!(toniefile.is_ok());
    /// # }
    /// ```
    pub fn new_simple(writer: F) -> Result<Toniefile<F>, ToniefileError> {
        let audio_id = rand::thread_rng().gen::<u32>();
        Toniefile::new(writer, audio_id, None)
    }

    /// Create a new Toniefile with ogg audio stream id audio_id,
    /// header prefilled and first two ogg pages header and comments written.
    ///
    /// - pass in a writer that implements the Write and Seek traits. (e.g. File or Cursor)
    /// - `audio_id` can be any u32 and is used to identify the audio file on the Toniebox
    /// - `user_comments` is a vector of strings that will be written to the Ogg comment page
    ///
    /// # Errors
    ///
    /// Returns an error if the space for comments in the file would overflow. (382 bytes, but
    /// every comment needs 4 bytes extra for the length indicator)
    /// returns a Toniefile struct if successful
    ///
    /// # Example
    ///
    /// ```
    /// # use std::io::Cursor;
    /// # use toniefile::Toniefile;
    /// #
    /// # fn create_toniefile() {
    ///     let myvec: Vec<u8> = vec![];
    ///     let cursor = Cursor::new(myvec);
    ///     let toniefile = Toniefile::new(cursor, 0x12345678, Some(vec!["my comment", "another comment"]));
    ///     assert!(toniefile.is_ok());
    /// # }
    /// ```
    pub fn new(
        writer: F,
        audio_id: u32,
        user_comments: Option<Vec<&str>>,
    ) -> Result<Toniefile<F>, ToniefileError> {
        let header = TonieboxAudioFileHeader {
            audio_id,
            num_bytes: TONIE_LENGTH_MAX as u64,
            track_page_nums: vec![],
            ..Default::default()
        };

        let mut toniefile = Toniefile {
            writer,
            file_position: 0,
            audio_length: 0,
            opus_encoder: OpusEnc::new(
                SampleRate::Hz48000,
                audiopus::Channels::Stereo,
                audiopus::Application::Audio,
            )?,
            audio_frame: [0; OPUS_FRAME_SIZE * OPUS_CHANNELS],
            audio_frame_used: 0,
            ogg_stream: OggStream::new(audio_id as i32),
            ogg_granulepos: 0,
            ogg_packet_count: 0,
            taf_header: header,
            taf_page_number: 0,
            sha1_ctx: Sha1::new(),
        };
        toniefile.new_chapter()?;

        toniefile.write_header()?;
        toniefile
            .writer
            .seek(SeekFrom::Start(TONIEFILE_FRAME_SIZE as u64))?; // seek after header

        // opus settings
        toniefile
            .opus_encoder
            .set_bitrate(Bitrate::BitsPerSecond(OPUS_BITRATE as i32))?;
        toniefile.opus_encoder.set_vbr(true)?;
        toniefile.opus_encoder.set_encoder_ctl_request(
            ffi::OPUS_SET_EXPERT_FRAME_DURATION_REQUEST,
            OPUS_FRAME_SIZE_MS_OPUS,
        )?;
        #[rustfmt::skip]
        let opus_header: [u8; 19] = [
            b'O', b'p', b'u', b's', b'H', b'e', b'a', b'd',                      // magic
            0x01,                                                                // opus version
            OPUS_CHANNELS as u8,                                                 // channel count
            0x38, 0x01,                                                          // pre-skip
            (OPUS_SAMPLE_RATE as u8), (OPUS_SAMPLE_RATE >> 8) as u8, 0x00, 0x00, // input sample rate
            0x00, 0x00,                                                          // output gain
            0x00,                                                                // channel map
        ];

        let mut opus_tags: [u8; COMMENT_LEN] = [b'0'; COMMENT_LEN];
        let mut tags_cursor = Cursor::new(&mut opus_tags[..]);
        let _ = tags_cursor.write(b"OpusTags")?;
        toniefile.comment_add(
            &mut tags_cursor,
            &format!("Rust toniefile encoder {}", std::env!("CARGO_PKG_VERSION")),
        )?;
        // NOTE: use this when audiopus 0.3 is stable
        // let libopusversion = format!("opus {}", audiopus::version()); // this is the libopus version not the audiopus version
        let libopusversion = unsafe { CStr::from_ptr(ffi::opus_get_version_string()) }.to_str()?;
        toniefile.comment_add(&mut tags_cursor, libopusversion)?;
        if let Some(user_comments) = user_comments {
            for user_comment in user_comments {
                toniefile.comment_add(&mut tags_cursor, user_comment)?;
            }
        }
        if (tags_cursor.position() as usize) < COMMENT_LEN {
            let mut len_bf = [0u8; 4];
            LittleEndian::write_u32(
                &mut len_bf,
                COMMENT_LEN as u32 - tags_cursor.position() as u32 - 4,
            );
            let _ = tags_cursor.write(&len_bf)?;
        }

        let mut header_packet = OggPacket::new(&opus_header);
        header_packet.set_bos(true);
        header_packet.set_eos(false);
        header_packet.set_granulepos(0);
        header_packet.set_packetno(toniefile.ogg_packet_count);
        toniefile.ogg_packet_count += 1;

        let mut comment_packet = OggPacket::new(&opus_tags);
        comment_packet.set_bos(false);
        comment_packet.set_eos(false);
        comment_packet.set_granulepos(0);
        comment_packet.set_packetno(toniefile.ogg_packet_count);
        toniefile.ogg_packet_count += 1;

        toniefile.ogg_stream.packetin(&mut header_packet);
        toniefile.ogg_stream.packetin(&mut comment_packet);

        toniefile.file_position = 0;

        while let Some(og) = toniefile.ogg_stream.flush() {
            toniefile.writer.write_all(&og.header)?;
            toniefile.writer.write_all(&og.body)?;
            toniefile.file_position += (og.header.len() + og.body.len()) as u64;
            toniefile.audio_length += (og.header.len() + og.body.len()) as u32;

            toniefile.sha1_ctx.update(&og.header);
            toniefile.sha1_ctx.update(&og.body);
        }

        Ok(toniefile)
    }

    /// Add a new chapter to the Toniefile's audio data
    ///
    /// # Errors
    ///
    /// Returns an error if the maximum number of chapters (100) is exceeded.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::io::{Cursor, Seek, SeekFrom, Write};
    /// # use toniefile::{Toniefile, ToniefileError};
    /// #
    /// # fn make_two_track_toniefile() -> Result<(), ToniefileError> {
    /// #    let myvec: Vec<u8> = vec![];
    /// #    let cursor = Cursor::new(myvec);
    ///     let mut toniefile = Toniefile::new(cursor, 0x12345678, None)?;
    ///     let samples: Vec<i16> = vec![0; 48000 * 2 * 60]; // 60 seconds of finest silence
    ///     let res = toniefile.encode(&samples);
    /// #    assert!(res.is_ok());
    ///
    ///     let res = toniefile.new_chapter();
    ///     assert!(res.is_ok());
    ///     let res = toniefile.encode(&samples); // encode another 60s of fine silence
    ///     assert!(res.is_ok());
    ///     toniefile.finalize_no_consume()?;
    ///
    ///     let header = Toniefile::parse_header(&mut toniefile.writer())?;
    ///     // the header should now contain two track_page_nums
    ///     assert_eq!(header.track_page_nums.len(), 2);
    /// #   Ok(())
    /// # }
    /// ```
    pub fn new_chapter(&mut self) -> Result<(), ToniefileError> {
        if self.taf_header.track_page_nums.len() > TONIEFILE_MAX_CHAPTERS {
            return Err(ToniefileError::MaxChaptersReached);
        }
        self.taf_header.track_page_nums.push(self.taf_page_number);
        Ok(())
    }

    /// Encode data into the Toniefile. This function can be called either once with a single buffer of
    /// samples or multiple times with smaller buffers. In the latter case, the buffers will be appended
    /// to the file.
    /// The samples must be interleaved, following the order: left, right, left, right, ...
    /// Additionally, the samples must be 16-bit signed integers.
    ///
    /// # Errors
    ///
    /// Returns an error if a misalignment happens during encoding.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::io::{Cursor, Seek, SeekFrom, Write};
    /// # use toniefile::{Toniefile, ToniefileError};
    /// #
    /// # fn fill_vector_toniefile() -> Result<(), ToniefileError> {
    /// #     let myvec: Vec<u8> = vec![];
    /// #     let cursor = Cursor::new(myvec);
    ///      let mut toniefile = Toniefile::new(cursor, 0x12345678, None)?;
    ///     let samples: Vec<i16> = vec![0; 48000 * 2 * 60]; // 60 seconds of finest silence
    ///
    ///     let res = toniefile.encode(&samples);
    ///     assert!(res.is_ok());
    ///
    /// #    toniefile.finalize_no_consume()?;
    /// #   Ok(())
    /// # }
    pub fn encode(&mut self, sample_buf: &[i16]) -> Result<(), ToniefileError> {
        // TODO get rid of samples available, use sample_buf.len()
        const PAGE_HEADER_SIZE: i64 = 27;
        let mut samples_processed = 0;
        let mut output_frame = [0u8; TONIEFILE_FRAME_SIZE];

        while samples_processed < sample_buf.len() / OPUS_CHANNELS {
            let mut samples = OPUS_FRAME_SIZE - self.audio_frame_used;
            let samples_remaining = sample_buf.len() / OPUS_CHANNELS - samples_processed;
            if samples > samples_remaining {
                samples = samples_remaining;
            }

            // this looks awful but it is basically only copying a slice of samples into audio_frame
            // maybe use some kind of windowing iterator instead?
            self.audio_frame[self.audio_frame_used * OPUS_CHANNELS
                ..(self.audio_frame_used * OPUS_CHANNELS + samples * OPUS_CHANNELS)]
                .copy_from_slice(
                    &sample_buf[samples_processed * OPUS_CHANNELS
                        ..(samples_processed * OPUS_CHANNELS + samples * OPUS_CHANNELS)],
                );
            self.audio_frame_used += samples;
            samples_processed += samples;

            if self.audio_frame_used < OPUS_FRAME_SIZE {
                continue;
            }
            let mut page_used = (self.file_position % TONIEFILE_FRAME_SIZE as u64) as i64
                + PAGE_HEADER_SIZE
                + self.ogg_stream.get_lacing_fill()
                - self.ogg_stream.get_lacing_returned()
                + self.ogg_stream.get_body_fill()
                - self.ogg_stream.get_body_returned();
            let mut page_remain = TONIEFILE_FRAME_SIZE as i64 - page_used;
            let mut frame_payload = (page_remain / 256) * 255 + (page_remain % 256) - 1;
            let reconstructed = (frame_payload / 255) + 1 + frame_payload;

            // when due to segment sizes we would end up with a 1 byte gap, make sure that the next run will have at least 64 byte.
            // reason why this could happen is that "adding one byte" would require one segment more and thus occupies two byte more.
            // if this would happen, just reduce the calculated free space such that there is room for another segment.
            if page_remain != reconstructed && frame_payload > OPUS_PACKET_MINSIZE as i64 {
                frame_payload -= OPUS_PACKET_MINSIZE as i64
            }
            if frame_payload < OPUS_PACKET_MINSIZE as i64 {
                return Err(ToniefileError::NotEnoughSpace);
            }

            let mut frame_len = self.opus_encoder.encode(
                &self.audio_frame[..2 * OPUS_FRAME_SIZE],
                &mut output_frame[..frame_payload as usize],
            )?;
            if frame_payload - (frame_len as i64) < (OPUS_PACKET_PAD as i64) {
                let target_length = frame_payload;
                packet_pad(&mut output_frame[..frame_len], target_length as i32)?;
                frame_len = target_length as usize;
            }

            // we have to retrieve the actually encoded samples in this frame
            let nb_frames;
            unsafe {
                // NOTE: get_nb_frames is not available in audiopus 0.2. Once 0.3 is released, this can be removed
                // and the safe getter function can be used instead.
                nb_frames = ffi::opus_packet_get_nb_frames(output_frame.as_ptr(), frame_len as i32);
            }
            let frames =
                samples_per_frame(&output_frame[..], SampleRate::Hz48000)? * nb_frames as usize;
            if frames != OPUS_FRAME_SIZE {
                return Err(ToniefileError::FrameSizeDontMatch(frames, OPUS_FRAME_SIZE));
            }
            self.ogg_granulepos += frames as u64;

            let output_vec = &output_frame[..frame_len]; // TODO wtf? why can I not pass in the slice directly?
            let mut packet = OggPacket::new(&output_vec);
            packet.set_bos(false);
            packet.set_eos(false);
            packet.set_granulepos(self.ogg_granulepos as i64);
            packet.set_packetno(self.ogg_packet_count);

            self.ogg_packet_count += 1;

            self.ogg_stream.packetin(&mut packet);

            page_used = (self.file_position % TONIEFILE_FRAME_SIZE as u64) as i64
                + PAGE_HEADER_SIZE
                + self.ogg_stream.get_lacing_fill()
                + self.ogg_stream.get_body_fill();
            page_remain = TONIEFILE_FRAME_SIZE as i64 - page_used;

            if page_remain < TONIEFILE_PAD_END as i64 {
                if page_remain > 0 {
                    return Err(ToniefileError::SmallPaddingError(
                        self.ogg_granulepos,
                        self.ogg_granulepos / OPUS_FRAME_SIZE as u64 * 60 / 1000,
                    ));
                }

                while let Some(og) = self.ogg_stream.flush() {
                    self.writer.write_all(&og.header)?;
                    self.writer.write_all(&og.body)?;
                    let prev = self.file_position;
                    self.file_position += (og.header.len() + og.body.len()) as u64;
                    self.audio_length += (og.header.len() + og.body.len()) as u32;

                    self.sha1_ctx.update(&og.header);
                    self.sha1_ctx.update(&og.body);

                    if prev / TONIEFILE_FRAME_SIZE as u64
                        != self.file_position / TONIEFILE_FRAME_SIZE as u64
                    {
                        self.taf_page_number += 1;
                        if self.file_position % TONIEFILE_FRAME_SIZE as u64 != 0 {
                            return Err(ToniefileError::BlockAlignmentError(self.file_position));
                        }
                    }
                }
            }
            // fill again
            self.audio_frame_used = 0;
        }
        Ok(())
    }

    /// Finalize the Toniefile by computing and writing the SHA-1 hash in the header.
    /// This operation consumes the Toniefile struct and can only be called once.
    /// The Toniefile is dropped, automatically closing any open files.
    ///
    /// # Errors
    ///
    /// Returns an error if the SHA-1 hash could not be computed or the header could not be written.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::io::{Cursor, Seek, SeekFrom, Write};
    /// # use toniefile::{Toniefile, ToniefileError};
    /// #
    /// # fn test_finalize() -> Result<(), ToniefileError> {
    /// #     let myvec: Vec<u8> = vec![];
    /// #     let cursor = Cursor::new(myvec);
    ///     let mut toniefile = Toniefile::new(cursor, 0x12345678, None)?;
    ///     let samples: Vec<i16> = vec![0; 48000 * 2 * 60]; // 60 seconds of finest silence
    /// #
    ///     let res = toniefile.encode(&samples);
    /// #    assert!(res.is_ok());
    /// #
    ///     assert!(toniefile.finalize().is_ok());
    ///     // after finalize() toniefile is dropped
    /// #   Ok(())
    /// # }
    /// ```
    pub fn finalize(mut self) -> Result<(), ToniefileError> {
        self.writer.flush()?;
        self.taf_header.sha1_hash = self.sha1_ctx.finalize_fixed_reset().to_vec();
        self.taf_header.num_bytes = self.audio_length as u64;

        self.write_header()?;

        Ok(())
    }

    /// Finalize the Toniefile by writing the SHA-1 hash in the header.
    /// This method does not consume the Toniefile struct; however, open file
    /// descriptors are not closed.
    /// Use this method when you want to write the Toniefile to a vector, so the
    /// cursor-wrapped vector at `self.writer` can be used later.
    ///
    /// # Errors
    ///
    /// Returns an error if the SHA-1 hash could not be computed or the header could not be written.
    ///
    /// # Example
    ///
    /// ```
    /// # use std::io::{Cursor, Seek, SeekFrom, Write};
    /// # use toniefile::{Toniefile, ToniefileError};
    /// #
    /// # fn test_finalize_no_consume() {
    /// #     let myvec: Vec<u8> = vec![];
    /// #     let cursor = Cursor::new(myvec);
    ///     let mut toniefile = Toniefile::new(cursor, 0x12345678, None).unwrap();
    ///     let samples: Vec<i16> = vec![0; 48000 * 2 * 60]; // 60 seconds of finest silence
    /// #
    ///     let res = toniefile.encode(&samples);
    ///     assert!(res.is_ok());
    /// #
    ///     toniefile.finalize_no_consume().unwrap();
    ///     let mut new_cursor = toniefile.writer();
    ///     // finalize_no_consume automatically rewinds the cursor
    ///     assert_eq!(new_cursor.position(), 0);
    /// # }
    /// ```
    pub fn finalize_no_consume(&mut self) -> Result<(), ToniefileError> {
        self.writer.flush()?;
        self.taf_header.sha1_hash = self.sha1_ctx.finalize_fixed_reset().to_vec();
        self.taf_header.num_bytes = self.audio_length as u64;

        self.write_header()?;
        self.writer.rewind()?;

        Ok(())
    }

    /// Consume the Toniefile struct and return its writer.
    /// Only call this after calling finalize_no_consume() to retrieve the writer.
    /// ```
    /// # use std::io::{Cursor, Seek, SeekFrom, Write};
    /// # use toniefile::{Toniefile, ToniefileError};
    /// #
    /// # fn test_get_writer() -> Result<(), ToniefileError> {
    /// #     let myvec: Vec<u8> = vec![];
    /// #     let cursor = Cursor::new(myvec);
    /// #     let mut toniefile = Toniefile::new(cursor, 0x12345678, None)?;
    /// #     let samples: Vec<i16> = vec![0; 48000 * 2 * 60]; // 60 seconds of finest silence
    /// #
    /// #     let res = toniefile.encode(&samples);
    /// #     assert!(res.is_ok());
    /// #
    ///     toniefile.finalize_no_consume()?;
    ///     let mut new_cursor = toniefile.writer();
    ///     // finalize_no_consume automatically rewinds the cursor
    ///     assert_eq!(new_cursor.position(), 0);
    /// #   Ok(())
    /// # }
    /// ```
    pub fn writer(self) -> F {
        self.writer
    }

    /// Get a reference to the header of the Toniefile struct.
    /// This method allows access to the header data during the creation of the Toniefile,
    /// but note that the header is only updated during new() / new_simple() and  when
    /// finalize() or finalize_no_consume() is called.
    /// ```
    /// # use std::io::{Cursor, Seek, SeekFrom, Write};
    /// # use toniefile::{Toniefile, ToniefileError};
    /// #
    /// # fn test_get_header() -> Result<(), ToniefileError> {
    /// #    let myvec: Vec<u8> = vec![];
    /// #    let cursor = Cursor::new(myvec);
    ///     let mut toniefile = Toniefile::new(cursor, 0x12345678, None)?;
    ///     let header = toniefile.header();
    ///     assert_eq!(header.audio_id, 0x12345678);
    /// #   Ok(())
    /// # }
    /// ```
    pub fn header(&self) -> &TonieboxAudioFileHeader {
        &self.taf_header
    }

    /// Get the current audio length in bytes.
    /// ```
    /// # use std::io::{Cursor, Seek, SeekFrom, Write};
    /// # use toniefile::{Toniefile, ToniefileError};
    /// #
    /// # fn test_get_header() -> Result<(), ToniefileError> {
    /// #    let myvec: Vec<u8> = vec![];
    /// #    let cursor = Cursor::new(myvec);
    ///     let mut toniefile = Toniefile::new(cursor, 0x12345678, None)?;
    ///     let audio_length = toniefile.audio_length();
    ///     // after creation the audio length is 0
    ///     assert_eq!(audio_length, 0);
    /// #   Ok(())
    /// # }
    /// ```
    pub fn audio_length(&self) -> u32 {
        self.audio_length
    }

    /// Get the current file position in bytes.
    /// ```
    /// # use std::io::{Cursor, Seek, SeekFrom, Write};
    /// # use toniefile::{Toniefile, ToniefileError};
    /// #
    /// # fn test_get_header() -> Result<(), ToniefileError> {
    /// #    let myvec: Vec<u8> = vec![];
    /// #    let cursor = Cursor::new(myvec);
    ///     let mut toniefile = Toniefile::new(cursor, 0x12345678, None)?;
    ///     let file_position = toniefile.file_position();
    ///     // after creation the file position is at the end of the Ogg Opus Comment page
    ///     assert_eq!(file_position, 0x1200);
    /// #   Ok(())
    /// # }
    /// ```
    pub fn file_position(&self) -> u64 {
        self.file_position
    }

    /// Get the current granule position.
    /// ```
    /// # use std::io::{Cursor, Seek, SeekFrom, Write};
    /// # use toniefile::{Toniefile, ToniefileError};
    /// #
    /// # fn test_get_header() -> Result<(), ToniefileError> {
    /// #    let myvec: Vec<u8> = vec![];
    /// #    let cursor = Cursor::new(myvec);
    ///     let mut toniefile = Toniefile::new(cursor, 0x12345678, None)?;
    ///     let granule_position = toniefile.granule_position();
    ///     // after creation the granule position is 0
    ///     assert_eq!(granule_position, 0);
    /// #   Ok(())
    /// # }
    /// ```
    pub fn granule_position(&self) -> u64 {
        self.ogg_granulepos
    }

    /// Get the current packet count.
    /// ```
    /// # use std::io::{Cursor, Seek, SeekFrom, Write};
    /// # use toniefile::{Toniefile, ToniefileError};
    /// #
    /// # fn test_get_header() -> Result<(), ToniefileError> {
    /// #    let myvec: Vec<u8> = vec![];
    /// #    let cursor = Cursor::new(myvec);
    ///     let mut toniefile = Toniefile::new(cursor, 0x12345678, None)?;
    ///     let packet_count = toniefile.packet_count();
    ///     // after creation the packet count is 2 (Ogg header and Ogg comment)
    ///     assert_eq!(packet_count, 2);
    /// #   Ok(())
    /// # }
    /// ```
    pub fn packet_count(&self) -> i64 {
        self.ogg_packet_count
    }

    /// Get the current page number.
    /// ```
    /// # use std::io::{Cursor, Seek, SeekFrom, Write};
    /// # use toniefile::{Toniefile, ToniefileError};
    /// #
    /// # fn test_get_header() -> Result<(), ToniefileError> {
    /// #    let myvec: Vec<u8> = vec![];
    /// #    let cursor = Cursor::new(myvec);
    ///     let mut toniefile = Toniefile::new(cursor, 0x12345678, None)?;
    ///     let page_number = toniefile.page_number();
    ///     // after creation the page number is 0
    ///     assert_eq!(page_number, 0);
    /// #   Ok(())
    /// # }
    /// ```
    pub fn page_number(&self) -> u32 {
        self.taf_page_number
    }

    // adds a comment to the ogg comment page. This can only be called during creation
    // otherwise it will mess up the lengths
    fn comment_add(
        &mut self,
        cursor: &mut Cursor<&mut [u8]>,
        comment: &str,
    ) -> Result<(), ToniefileError> {
        const LENGTH_LEN: usize = 4; // length of the string length indicator
                                     // need at least 2 * LENGTH_LEN, because we also need to write the length of the padding
        if 2 * LENGTH_LEN + comment.len() + cursor.position() as usize > COMMENT_LEN {
            return Err(ToniefileError::CommentWontFit(
                comment.len(),
                cursor.position() as usize,
                COMMENT_LEN,
            ));
        }

        let mut len_bf = [0u8; 4];
        LittleEndian::write_u32(&mut len_bf, comment.len() as u32);
        let _ = cursor.write(&len_bf)?;
        let _ = cursor.write(comment.as_bytes())?;
        Ok(())
    }

    // serializes the header to vector pointed to by buf
    // Toniefileheaders must be exactly 4092 bytes long. This function
    // creates a header of the correct size by filling the fill field of the
    // protobuf
    // writes the header into buf and returns the length of the serialized header
    fn init_header(&mut self) -> Result<(usize, Vec<u8>), ToniefileError> {
        let proto_frame_size: i16 = TONIEFILE_FRAME_SIZE as i16 - 4;

        // only fill the hash at the first time initializing the header
        if self.taf_header.sha1_hash.is_empty() {
            self.taf_header.sha1_hash = vec![0xFFu8; SHA1_DIGEST_SIZE];
        }
        self.taf_header.fill = vec![];
        let mut data_length = self.taf_header.encoded_len();

        if data_length < proto_frame_size as usize {
            self.taf_header.fill = vec![0u8; proto_frame_size as usize - data_length - 1];
            // NOTE -1 because byte 0 of fill
        }
        data_length = self.taf_header.encoded_len();

        let mut buf = vec![];
        self.taf_header.encode(&mut buf)?;

        assert_eq!(data_length, proto_frame_size as usize); // TODO remove for lib
        Ok((data_length, buf))
    }

    // write header to writer
    fn write_header(&mut self) -> Result<(), ToniefileError> {
        let mut len_bf = [0u8; 4];

        let (proto_size, buffer) = self.init_header()?;
        BigEndian::write_u32(&mut len_bf, proto_size as u32); // TODO can't we inline this?

        self.writer.seek(SeekFrom::Start(0))?;
        self.writer.write_all(&len_bf)?;

        self.writer.seek(SeekFrom::Start(4))?;
        self.writer.write_all(&buffer)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn get_test_assets_path() -> PathBuf {
        let test_path = std::env!("CARGO_MANIFEST_DIR")
            .parse::<PathBuf>()
            .unwrap()
            .join("test/assets");
        std::fs::create_dir_all(&test_path).unwrap();
        test_path
    }

    fn get_test_out_path() -> PathBuf {
        let test_path = std::env!("CARGO_MANIFEST_DIR")
            .parse::<PathBuf>()
            .unwrap()
            .join("test/out");
        std::fs::create_dir_all(&test_path).unwrap();
        test_path
    }

    fn read_file_i16(path: &str) -> Vec<i16> {
        let mut f = File::open(path).expect("no file found");
        let (_, b) = wav::read(&mut f).unwrap();
        b.try_into_sixteen().unwrap()
    }

    fn check_file_against_header<R: Read + Seek>(reader: &mut R) {
        let header = Toniefile::parse_header(reader).unwrap();
        reader.seek(SeekFrom::Start(0x1000)).unwrap();
        let mut output_buffer = vec![];
        reader.read_to_end(&mut output_buffer).unwrap();

        let mut hasher = Sha1::new();
        hasher.update(&output_buffer);
        let output_buffer_hash = hasher.finalize().to_vec();

        let audio_id = LittleEndian::read_u32(&output_buffer[14..=17]);
        assert_eq!(audio_id, header.audio_id);
        assert_eq!(output_buffer.len(), header.num_bytes as usize);
        assert_eq!(output_buffer_hash, header.sha1_hash);
    }

    #[test]
    fn create_toniefile() {
        let myvec: Vec<u8> = vec![];
        let cursor = Cursor::new(myvec);
        let toniefile =
            Toniefile::new(cursor, 0x12345678, Some(vec!["Hello World", "How are You"]));
        assert!(toniefile.is_ok());
    }

    #[test]
    fn create_simple_toniefile() {
        let myvec: Vec<u8> = vec![];
        let cursor = Cursor::new(myvec);
        let toniefile = Toniefile::new_simple(cursor);
        assert!(toniefile.is_ok());
    }

    #[test]
    #[ignore]
    fn just_enough_comments() {
        // NOTE: This test will fail on some machines because different infos about the environment are put into the headers
        // therefore a changing environment can cause a different length of the padding.
        let myvec: Vec<u8> = vec![];
        let cursor = Cursor::new(myvec);
        let comments = vec!["A"; 75]; // be aware that every comment adds 5 bytes to the header
        let toniefile = Toniefile::new(cursor, 0x12345678, Some(comments));
        assert!(toniefile.is_ok());
    }
    #[test]
    #[ignore]
    fn comment_padding_len() {
        // NOTE: This test will fail on some machines because different infos about the environment are put into the headers
        // therefore a changing environment can cause a different length of the padding.
        let myvec: Vec<u8> = vec![];
        let cursor = Cursor::new(myvec);
        // this should result in a padding of 7 bytes
        let comments = vec!["A"; 74]; // be aware that every comment adds 5 bytes to the header
        let toniefile = Toniefile::new(cursor, 0x12345678, Some(comments));
        assert!(toniefile.is_ok());
        let mut cursor = toniefile.unwrap().writer();
        // so length should be written 11 bytes from the end (7 for the padding and 4 for the length itself)
        cursor.seek(SeekFrom::Start(0x1200 - 0xB)).unwrap();
        let buf = &mut [0u8; 1];
        let _ = cursor.read_exact(buf);
        assert_eq!(buf[0], 7);
    }

    #[test]
    #[should_panic]
    fn too_many_comments() {
        // NOTE: Since the comment buffer is 436 bytes long and a single character comment takes 5 bytes space adding 100 "A"
        // comments (=500 bytes) will always panic, regardless of the environment and what is in the buffer before adding user comments.
        let myvec: Vec<u8> = vec![];
        let cursor = Cursor::new(myvec);
        let comments = vec!["A"; 100]; // be aware that every comment adds 5 bytes to the header
        let toniefile = Toniefile::new(cursor, 0x12345678, Some(comments));
        assert!(toniefile.is_ok());
    }

    #[test]
    fn fill_single_buffer_toniefile() {
        let file = File::create(get_test_out_path().join("500304E0")).unwrap();
        let mut toniefile = Toniefile::new(file, 0x12345678, None).unwrap();
        let samples: Vec<i16> =
            read_file_i16(get_test_assets_path().join("1000hz.wav").to_str().unwrap());
        let res = toniefile.encode(&samples);
        assert!(res.is_ok());

        toniefile.finalize().unwrap();
        let mut file = File::open(get_test_out_path().join("500304E0")).unwrap();
        check_file_against_header(&mut file);
    }

    #[test]
    fn fill_vector_toniefile() {
        let myvec: Vec<u8> = vec![];
        let cursor = Cursor::new(myvec);
        let mut toniefile = Toniefile::new(cursor, 0x12345678, None).unwrap();
        let samples: Vec<i16> =
            read_file_i16(get_test_assets_path().join("1000hz.wav").to_str().unwrap());
        let res = toniefile.encode(&samples);
        assert!(res.is_ok());

        toniefile.finalize_no_consume().unwrap();
        check_file_against_header(&mut toniefile.writer);
    }
    #[test]
    fn check_page_positions_toniefile() {
        let myvec: Vec<u8> = vec![];
        let cursor = Cursor::new(myvec);
        let mut toniefile = Toniefile::new(cursor, 0x12345678, None).unwrap();
        let samples: Vec<i16> =
            read_file_i16(get_test_assets_path().join("1000hz.wav").to_str().unwrap());
        let res = toniefile.encode(&samples);
        assert!(res.is_ok());

        toniefile.finalize_no_consume().unwrap();

        let mut reader = toniefile.writer();
        reader.seek(SeekFrom::Start(0x1000)).unwrap();
        let mut output_buffer = vec![];
        reader.read_to_end(&mut output_buffer).unwrap();

        for window in output_buffer.chunks(TONIEFILE_FRAME_SIZE) {
            assert!(window.starts_with(b"OggS"));
        }
    }

    #[test]
    fn fill_small_buffers_toniefile() {
        let myvec: Vec<u8> = vec![];
        let cursor = Cursor::new(myvec);
        let mut toniefile = Toniefile::new(cursor, 0x12345678, None).unwrap();

        let samples: Vec<i16> =
            read_file_i16(get_test_assets_path().join("1000hz.wav").to_str().unwrap());
        for window in samples.chunks(TONIEFILE_FRAME_SIZE * OPUS_CHANNELS) {
            let res = toniefile.encode(window);
            assert!(res.is_ok());
        }

        toniefile.finalize_no_consume().unwrap();
        check_file_against_header(&mut toniefile.writer);
    }

    #[test]
    fn read_and_fill_chunks_toniefile() {
        let myvec: Vec<u8> = vec![];
        let cursor = Cursor::new(myvec);
        let mut toniefile = Toniefile::new(cursor, 0x12345678, None).unwrap();

        let mut f =
            File::open(get_test_assets_path().join("1000hz.wav").to_str().unwrap()).unwrap();
        let mut wav_reader = hound::WavReader::new(&mut f).unwrap();
        let total_samples = wav_reader.duration();
        let mut wav_iter = wav_reader.samples::<i16>();
        let mut samples_read = 0;
        while samples_read < total_samples {
            let mut window = vec![];
            for _ in 0..TONIEFILE_FRAME_SIZE * OPUS_CHANNELS {
                window.push(wav_iter.next().unwrap().unwrap());
            }
            let res = toniefile.encode(&window);
            assert!(res.is_ok());
            samples_read += window.len() as u32;
        }

        toniefile.finalize_no_consume().unwrap();
        check_file_against_header(&mut toniefile.writer);
    }

    #[test]
    fn header_is_correct() {
        let myvec: Vec<u8> = vec![];
        let cursor = Cursor::new(myvec);
        let mut toniefile = Toniefile::new(cursor, 0x12345678, None).unwrap();
        let samples: Vec<i16> =
            read_file_i16(get_test_assets_path().join("1000hz.wav").to_str().unwrap());
        let res = toniefile.encode(&samples);
        assert!(res.is_ok());

        toniefile.finalize_no_consume().unwrap();

        let mut cursor = toniefile.writer();
        let header = Toniefile::parse_header(&mut cursor);
        assert!(header.is_ok());
        let header = header.unwrap();
        cursor.seek(SeekFrom::Start(0x1000)).unwrap();
        let mut output_buffer = vec![];
        cursor.read_to_end(&mut output_buffer).unwrap();

        let mut hasher = Sha1::new();
        hasher.update(&output_buffer);
        let output_buffer_hash = hasher.finalize().to_vec();

        let audio_id = LittleEndian::read_u32(&output_buffer[14..=17]);
        assert_eq!(audio_id, header.audio_id);
        assert_eq!(output_buffer.len(), header.num_bytes as usize);
        assert_eq!(output_buffer_hash, header.sha1_hash);
    }
}
