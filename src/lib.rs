use audiopus::{
    coder::{Encoder as OpusEnc, GenericCtl},
    Bitrate, SampleRate, ffi,
};
use libogg::{ Packet as OggPacket, Stream as OggStream, Page as OggPage };
use prost::Message;
use sha1::{Digest, Sha1};
use std::{fs::File, error::Error, io::{ErrorKind, Write, Seek, SeekFrom}, iter::repeat};
use std::io::Cursor;
use toniehead::TonieboxAudioFileHeader;
use anyhow::Result; // TODO remove later on an do some proper error handling
use anyhow::anyhow;
use byteorder::{LittleEndian, ByteOrder};

const OPUS_FRAME_SIZE_MS: u32 = 60;
const OPUS_FRAME_SIZE_MS_OPUS: i32 = 5006;
const OPUS_SAMPLE_RATE: u32 = 48000;
const OPUS_BITRATE: u32 = 96000;
const OPUS_FRAME_SIZE: u32 = 5006; // This is a special value see opus_defines.h
const OPUS_CHANNELS: u32 = 2;
const OPUS_PACKET_PAD: usize = 64;
const OPUS_PACKET_MINSIZE: usize = 64;

const TONIEFILE_FRAME_SIZE: usize = 4096;
const TONIEFILE_MAX_CHAPTERS: usize = 100;
const TONIEFILE_PAD_END: usize = 64;

pub mod toniehead {
    include!(concat!(env!("OUT_DIR"), "/toniehead.rs"));
}

#[derive(Debug)]
struct Toniefile {
    path: String,
    file: File,
    file_position: u64,
    audio_length: u32,
    // opus
    opus_encoder: OpusEnc,
    audio_frame: [i16; (OPUS_FRAME_SIZE * OPUS_CHANNELS) as usize],
    audio_frame_used: isize,
    // ogg
    ogg_stream: OggStream,
    ogg_granulepos: u64,
    ogg_packet_count: i64,
    // header
    header: TonieboxAudioFileHeader,
    sha1: Sha1,
    taf_block_number: u32,
}

impl Toniefile {
    pub fn comment_add(&mut self, buffer: &mut Vec<u8>, comment: &str) {
        let mut len_bf = [0u8;4];
        LittleEndian::write_u32(&mut len_bf, comment.len() as u32);
        buffer.extend(&len_bf);
        buffer.extend(comment.bytes());
    }
    pub fn header(&mut self, buf: &[u8], taf_header: &TonieboxAudioFileHeader) {
        todo!();
    }
    pub fn write_header(&mut self) {
        todo!();
    }
    pub fn create(path: &str, audio_id: u32) -> Result<Toniefile> {
        let header = TonieboxAudioFileHeader {
            audio_id,
            ..Default::default()
        };
        let mut toniefile = Toniefile {
            path: path.to_string(),
            file: File::create(path).expect("Could not create file"),
            file_position: 0,
            audio_length: 0,
            opus_encoder: OpusEnc::new(SampleRate::Hz48000,  audiopus::Channels::Stereo, audiopus::Application::Audio).expect("Could not create opus encoder"),
            audio_frame: [0; (OPUS_FRAME_SIZE * OPUS_CHANNELS) as usize],
            audio_frame_used: 0,
            ogg_stream: OggStream::new(audio_id as i32),
            ogg_granulepos: 0,
            ogg_packet_count: 0,
            header,
            taf_block_number: 0,
            sha1: Sha1::new(),
        };

        // write_header(&toniefile); // TODO
        toniefile.file.seek(SeekFrom::Start(TONIEFILE_FRAME_SIZE as u64)).expect("Could not seek to frame size");

        // opus settings
        toniefile.opus_encoder.set_bitrate(Bitrate::BitsPerSecond(OPUS_BITRATE as i32)).expect("Could not set bitrate");
        toniefile.opus_encoder.set_vbr(false).expect("Could not set vbr");
        toniefile.opus_encoder.set_encoder_ctl_request(ffi::OPUS_SET_EXPERT_FRAME_DURATION_REQUEST, OPUS_FRAME_SIZE_MS_OPUS).expect("Could not set frame duration");

        let opus_header: [u8; 19] = [
            b'O', b'p', b'u', b's', b'H', b'e', b'a', b'd',
            0x01,
            OPUS_CHANNELS as u8,
            0x38, 0x01,
            (OPUS_SAMPLE_RATE as u8) & 0xFF, (OPUS_SAMPLE_RATE >> 8) as u8, 0x00, 0x00,
            0x00, 0x00,
            0x00,
        ];
        let mut opus_tags: Vec<u8> = Vec::with_capacity(0x1B4);
        opus_tags.extend(b"OpusTags");
        toniefile.comment_add(&mut opus_tags, &format!("toniefile {}", std::env!("CARGO_PKG_VERSION")));
        // TODO add some opus user comments here

        // add padding of the first block
        let padding_str = "pad=";
        let mut len_bf = [0u8;4];
        LittleEndian::write_u32(&mut len_bf, (opus_tags.capacity() - opus_tags.len() - 4) as u32);
        opus_tags.extend(&len_bf);
        opus_tags.extend(padding_str.bytes());
        opus_tags.extend(repeat(b'0').take(opus_tags.capacity() - opus_tags.len()));

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
            toniefile.file.write_all(&og.header).expect("Could not write ogg header");
            toniefile.file.write_all(&og.body).expect("Could not write ogg body");
            toniefile.file_position += (og.header.len() + og.body.len()) as u64;
            toniefile.audio_length += (og.header.len() + og.body.len()) as u32;

            toniefile.sha1.update(&og.header);
            toniefile.sha1.update(&og.body);
        }

        Ok(toniefile)

    }
    pub fn close(&mut self) {
        todo!();
    }
    pub fn new_chapter(&mut self) {
        todo!();
    }
    pub fn encode(&mut self, sample_buf: &[i16], samples_available: usize) {
        todo!();
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn get_test_path() -> PathBuf {
        let test_path = std::env!("CARGO_MANIFEST_DIR").parse::<PathBuf>().unwrap().join("test_assets");
        std::fs::create_dir_all(&test_path).unwrap();
        test_path
    }

    #[test]
    fn create_toniefile() {
        let toniefile = Toniefile::create(
            get_test_path().join("test.tonie").to_str().unwrap(),
            0x12345678,
        );
        println!("{:x?}", toniefile);
        assert!(toniefile.is_ok());
    }
}
