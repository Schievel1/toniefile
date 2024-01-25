use audiopus::packet::samples_per_frame;
use audiopus::repacketizer::packet_pad;
use audiopus::{coder::Encoder as OpusEnc, ffi, Bitrate, SampleRate};
use byteorder::{BigEndian, ByteOrder, LittleEndian};
use libogg::{Packet as OggPacket, Stream as OggStream};
use prost::Message;
use sha1::digest::FixedOutputReset;
use sha1::{Digest, Sha1};
use std::error::Error;
use std::ffi::CStr;
use std::{
    fs::File,
    io::{Seek, SeekFrom, Write},
    iter::repeat,
};
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

use thiserror::Error;
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
    #[error("Unexpected small padding at granule postion{0} ( {1}sec)")]
    SmallPaddingError(u64, u64),
    #[error("Block alignment error at position {:#010x}", .0)]
    BlockAlignmentError(u64),
    #[error(transparent)]
    OpusEncoderError(#[from] audiopus::Error),
    #[error(transparent)]
    IOError(#[from] std::io::Error),
}

pub mod toniehead {
    #![allow(non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/toniehead.rs"));
}

#[derive(Debug)]
pub struct Toniefile {
    pub path: String,
    file: File,
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
    pub header: TonieboxAudioFileHeader,
    sha1: Sha1,
    taf_block_number: u32,
}

impl Toniefile {
    /// Add a comment to the opus tags
    pub fn comment_add(
        &mut self,
        buffer: &mut Vec<u8>,
        comment: &str,
    ) -> Result<(), Box<dyn Error>> {
        if comment.len() + buffer.len() > COMMENT_LEN {
            return Err(Box::new(ToniefileError::CommentWontFit(
                comment.len(),
                buffer.len(),
                COMMENT_LEN,
            )));
        }

        let mut len_bf = [0u8; 4];
        LittleEndian::write_u32(&mut len_bf, comment.len() as u32);
        buffer.extend(&len_bf);
        buffer.extend(comment.bytes());
        Ok(())
    }
    /// serializes the header to vector pointed to by buf
    /// Toniefileheaders must be exactly 4092 bytes long. This function
    /// creates a header of the correct size by filling the fill field of the
    /// protobuf
    /// writes the header into buf and returns the length of the serialized header
    pub fn header(&mut self, buf: &mut Vec<u8>) -> Result<usize, Box<dyn Error>> {
        let proto_frame_size: i16 = TONIEFILE_FRAME_SIZE as i16 - 4;

        // only fill the hash at the first time initializing the header
        if self.header.sha1_hash.is_empty() {
            self.header.sha1_hash = vec![0xFFu8; SHA1_DIGEST_SIZE];
        }
        self.header.fill = vec![];
        let mut data_length = self.header.encoded_len();

        if data_length < proto_frame_size as usize {
            self.header.fill = vec![0u8; proto_frame_size as usize - data_length - 1];
            // NOTE -1 because byte 0 of fill
        }
        data_length = self.header.encoded_len();

        self.header.encode(buf)?;

        assert_eq!(data_length, proto_frame_size as usize); // TODO remove for lib
        Ok(data_length)
    }

    /// write header to writer
    pub fn write_header(&mut self) -> Result<(), Box<dyn Error>> {
        let mut buffer = vec![];
        let mut len_bf = [0u8; 4];

        let proto_size = self.header(&mut buffer)?;
        BigEndian::write_u32(&mut len_bf, proto_size as u32); // TODO can't we inline this?

        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&len_bf)?;

        self.file.seek(SeekFrom::Start(4))?;
        self.file.write_all(&buffer)?;
        Ok(())
    }

    /// Create a new Toniefile at path with ogg audio stream id audio_id with
    /// header prefilled and first two ogg pages header and comments written
    /// returns a Toniefile struct if successful
    pub fn create(
        path: &str,
        audio_id: u32,
        user_comment: &str,
    ) -> Result<Toniefile, Box<dyn Error>> {
        let header = TonieboxAudioFileHeader {
            audio_id,
            num_bytes: TONIE_LENGTH_MAX as u64,
            track_page_nums: vec![],
            ..Default::default()
        };

        let mut toniefile = Toniefile {
            path: path.to_string(),
            file: File::create(path)?,
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
            header,
            taf_block_number: 0,
            sha1: Sha1::new(),
        };
        let _ = toniefile.new_chapter();

        toniefile.write_header()?;
        toniefile
            .file
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

        let mut opus_tags: Vec<u8> = Vec::with_capacity(0x1B4);
        opus_tags.extend(b"OpusTags");
        toniefile.comment_add(
            &mut opus_tags,
            &format!("Rust toniefile encoder {}", std::env!("CARGO_PKG_VERSION")),
        )?;
        // NOTE: use this when audiopus 0.3 is stable
        // let libopusversion = format!("opus {}", audiopus::version()); // this is the libopus version not the audiopus version
        let libopusversion = unsafe { CStr::from_ptr(ffi::opus_get_version_string()) }.to_str()?;
        toniefile.comment_add(&mut opus_tags, libopusversion)?;
        if !user_comment.is_empty() {
            toniefile.comment_add(&mut opus_tags, user_comment)?;
        }

        // add padding of the first Ogg block
        if opus_tags.len() < opus_tags.capacity() - 5 {
            // there is room for a new string, add four legnth bytes as the string devider
            // and '0's for the rest of the string
            let mut len_bf = [0u8; 4];
            LittleEndian::write_u32(
                &mut len_bf,
                (opus_tags.capacity() - opus_tags.len() - 4) as u32,
            );
            opus_tags.extend(&len_bf);
            opus_tags.extend(repeat(b'0').take(opus_tags.capacity() - opus_tags.len()));
        } else if opus_tags.len() < opus_tags.capacity() {
            // if there is no room for a new string, just extent the last comment with '0'
            opus_tags.extend(repeat(b'0').take(opus_tags.capacity() - opus_tags.len()));
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
            toniefile.file.write_all(&og.header)?;
            toniefile.file.write_all(&og.body)?;
            toniefile.file_position += (og.header.len() + og.body.len()) as u64;
            toniefile.audio_length += (og.header.len() + og.body.len()) as u32;

            toniefile.sha1.update(&og.header);
            toniefile.sha1.update(&og.body);
        }

        Ok(toniefile)
    }

    /// Finalize the toniefile by writing the last ogg page and the sha1 hash
    /// in the header. This consumes the Toniefile struct and can therefore only
    /// be called once
    pub fn finalize(mut self) -> Result<(), Box<dyn Error>> {
        self.file.flush()?;
        self.header.sha1_hash = self.sha1.finalize_fixed_reset().to_vec();
        self.header.num_bytes = self.audio_length as u64;

        self.write_header()?;

        Ok(())
    }

    pub fn new_chapter(&mut self) -> Result<(), Box<dyn Error>> {
        if self.header.track_page_nums.len() > TONIEFILE_MAX_CHAPTERS {
            return Err(Box::new(ToniefileError::MaxChaptersReached));
        }
        self.header.track_page_nums.push(self.taf_block_number);
        Ok(())
    }

    pub fn encode(&mut self, sample_buf: &[i16]) -> Result<(), Box<dyn Error>> {
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

            /* when due to segment sizes we would end up with a 1 byte gap, make sure that the next run will have at least 64 byte.
             * reason why this could happen is that "adding one byte" would require one segment more and thus occupies two byte more.
             * if this would happen, just reduce the calculated free space such that there is room for another segment.
             */
            if page_remain != reconstructed && frame_payload > OPUS_PACKET_MINSIZE as i64 {
                frame_payload -= OPUS_PACKET_MINSIZE as i64
            }
            // println!("page remain {}", page_remain);
            // println!("frame payload {}", frame_payload);
            if frame_payload < OPUS_PACKET_MINSIZE as i64 {
                return Err(Box::new(ToniefileError::NotEnoughSpace));
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

            /* we have to retrieve the actually encoded samples in this frame */
            let nb_frames;
            unsafe {
                // NOTE: get_nb_frames is not available in audiopus 0.2. Once 0.3 is released, this can be removed
                // and the safe getter function can be used instead.
                nb_frames = ffi::opus_packet_get_nb_frames(output_frame.as_ptr(), frame_len as i32);
            }
            let frames =
                samples_per_frame(&output_frame[..], SampleRate::Hz48000)? * nb_frames as usize;
            if frames != OPUS_FRAME_SIZE {
                return Err(Box::new(ToniefileError::FrameSizeDontMatch(
                    frames,
                    OPUS_FRAME_SIZE,
                )));
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
                    return Err(Box::new(ToniefileError::SmallPaddingError(
                        self.ogg_granulepos,
                        self.ogg_granulepos / OPUS_FRAME_SIZE as u64 * 60 / 1000,
                    )));
                }

                while let Some(og) = self.ogg_stream.flush() {
                    self.file.write_all(&og.header)?;
                    self.file.write_all(&og.body)?;
                    let prev = self.file_position;
                    self.file_position += (og.header.len() + og.body.len()) as u64;
                    self.audio_length += (og.header.len() + og.body.len()) as u32;

                    self.sha1.update(&og.header);
                    self.sha1.update(&og.body);

                    if prev / TONIEFILE_FRAME_SIZE as u64
                        != self.file_position / TONIEFILE_FRAME_SIZE as u64
                    {
                        self.taf_block_number += 1;
                        if self.file_position % TONIEFILE_FRAME_SIZE as u64 != 0 {
                            return Err(Box::new(ToniefileError::BlockAlignmentError(
                                self.file_position,
                            )));
                        }
                    }
                }
            }
            // fill again
            self.audio_frame_used = 0;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn get_test_path() -> PathBuf {
        let test_path = std::env!("CARGO_MANIFEST_DIR")
            .parse::<PathBuf>()
            .unwrap()
            .join("test_assets");
        std::fs::create_dir_all(&test_path).unwrap();
        test_path
    }
    fn read_file_i16(path: &str) -> Vec<i16> {
        let mut f = File::open(path).expect("no file found");
        let (_, b) = wav::read(&mut f).unwrap();
        b.try_into_sixteen().unwrap()
    }

    #[test]
    fn create_toniefile() {
        let toniefile = Toniefile::create(
            get_test_path().join("test.tonie").to_str().unwrap(),
            0x12345678,
            "",
        );
        // println!("{:x?}", toniefile.as_mut().unwrap().file.stream_position().unwrap());
        assert!(toniefile.is_ok());
    }

    #[test]
    fn fill_single_buffer_toniefile() {
        let mut toniefile = Toniefile::create(
            get_test_path().join("500304E0").to_str().unwrap(),
            0x12345678,
            "",
        )
        .unwrap();
        let samples: Vec<i16> = read_file_i16(get_test_path().join("1000hz.wav").to_str().unwrap());
        let res = toniefile.encode(&samples);
        assert!(res.is_ok());

        toniefile.finalize().unwrap();

        // let header_output_len = toniefile.header.num_bytes;
        // let header_hash = toniefile.header.sha1_hash.clone();

        // let mut output_file =
        //     File::open(get_test_path().join("500304E0").to_str().unwrap()).unwrap();
        // output_file.seek(SeekFrom::Start(0x1000)).unwrap();
        // let mut output_buffer = vec![];
        // output_file.read_to_end(&mut output_buffer).unwrap();

        // let mut hasher = Sha1::new();
        // hasher.update(&output_buffer);
        // let hash = hasher.finalize().to_vec();

        // assert_eq!(output_buffer.len(), header_output_len as usize);
        // assert_eq!(hash, header_hash);
    }

    #[test]
    fn fill_small_buffers_toniefile() {
        let mut toniefile = Toniefile::create(
            get_test_path().join("500304E0").to_str().unwrap(),
            0x12345678,
            "",
        )
        .unwrap();

        let samples: Vec<i16> = read_file_i16(get_test_path().join("1000hz.wav").to_str().unwrap());
        for window in samples.chunks(TONIEFILE_FRAME_SIZE * OPUS_CHANNELS) {
            let res = toniefile.encode(window);
            assert!(res.is_ok());
        }

        toniefile.finalize().unwrap();

        // let header_output_len = toniefile.header.num_bytes;
        // let header_hash = toniefile.header.sha1_hash.clone();

        // let mut output_file =
        //     File::open(get_test_path().join("500304E0").to_str().unwrap()).unwrap();
        // output_file.seek(SeekFrom::Start(0x1000)).unwrap();
        // let mut output_buffer = vec![];
        // output_file.read_to_end(&mut output_buffer).unwrap();

        // let mut hasher = Sha1::new();
        // hasher.update(&output_buffer);
        // let hash = hasher.finalize().to_vec();

        // assert_eq!(output_buffer.len(), header_output_len as usize);
        // assert_eq!(hash, header_hash);
    }
}
