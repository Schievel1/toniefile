use anyhow::anyhow;
use anyhow::Result; use audiopus::packet::Packet;
// TODO remove later on an do some proper error handling
use audiopus::{
    coder::{Encoder as OpusEnc, GenericCtl},
    ffi, Bitrate, SampleRate,
};
use byteorder::{BigEndian, ByteOrder, LittleEndian};
use libogg::{Packet as OggPacket, Page as OggPage, Stream as OggStream};
use prost::Message;
use sha1::{Digest, Sha1};
use std::io::Cursor;
use std::{
    error::Error,
    fs::File,
    io::{ErrorKind, Seek, SeekFrom, Write},
    iter::repeat,
};
use toniehead::TonieboxAudioFileHeader;
use audiopus::packet::samples_per_frame;
use audiopus::repacketizer::packet_pad;


const OPUS_FRAME_SIZE_MS: u32 = 60;
const OPUS_FRAME_SIZE_MS_OPUS: i32 = 5006; // This is a special value see opus_defines.h
const OPUS_SAMPLE_RATE: u32 = 48000;
const OPUS_BITRATE: u32 = 96000;
const OPUS_FRAME_SIZE: u32 = OPUS_SAMPLE_RATE * OPUS_FRAME_SIZE_MS / 1000;
const OPUS_CHANNELS: u32 = 2;
const OPUS_PACKET_PAD: usize = 64;
const OPUS_PACKET_MINSIZE: usize = 64;

const TONIEFILE_FRAME_SIZE: usize = 4096;
const TONIEFILE_MAX_CHAPTERS: usize = 100;
const TONIEFILE_PAD_END: usize = 64;
const CONTENT_LENGTH_MAX: i32 = i32::MAX;
const TONIE_LENGTH_MAX: i32 = CONTENT_LENGTH_MAX - 0x1000;

const SHA1_DIGEST_SIZE: usize = 160;

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
    n_taf_blocks: usize,
}

impl Toniefile {
    /// Add a comment to the opus tags
    pub fn comment_add(&mut self, buffer: &mut Vec<u8>, comment: &str) {
        let mut len_bf = [0u8; 4];
        LittleEndian::write_u32(&mut len_bf, comment.len() as u32);
        buffer.extend(&len_bf);
        buffer.extend(comment.bytes());
    }
    /// serializes the header to vector pointed to by buf
    /// Toniefileheaders must be exactly 4092 bytes long. This function
    /// creates a header of the correct size by filling the fill field of the
    /// protobuf
    /// returns the length of the serialized header
    pub fn header(&mut self, buf: &mut Vec<u8>) -> usize {
        let proto_frame_size: i16 = TONIEFILE_FRAME_SIZE as i16 - 4;

        self.header.sha1_hash = vec![0xFFu8; SHA1_DIGEST_SIZE];

        let mut data_length = self.header.encoded_len();
        println!("data length: {}", data_length);
        println!("proto frame size: {}", proto_frame_size);
        if data_length < proto_frame_size as usize {
            self.header.fill = vec![0u8; proto_frame_size as usize - data_length - 1]; // NOTE -1 because byte 0 of fill
        }
        data_length = self.header.encoded_len();

        self.header.encode(buf).expect("Could not encode header");
        data_length
    }

    /// write header to file
    pub fn write_header(&mut self) {
        let mut buffer = vec![];
        let mut len_bf = [0u8; 4];

        let proto_size = self.header(&mut buffer);
        BigEndian::write_u32(&mut len_bf, proto_size as u32); // TODO can't we inline this?

        self.file
            .seek(SeekFrom::Start(0))
            .expect("Could not seek to header");
        self.file
            .write_all(&len_bf)
            .expect("Could not write header size");

        self.file
            .seek(SeekFrom::Start(4))
            .expect("Could not seek to header"); // TODO is this necessary?
        self.file
            .write_all(&buffer)
            .expect("Could not write header");
    }

    /// Create a new Toniefile at path with ogg audio stream id audio_id with
    /// header prefilled and first two ogg pages header and comments written
    /// returns a Toniefile struct if successful
    pub fn create(path: &str, audio_id: u32) -> Result<Toniefile> {
        let header = TonieboxAudioFileHeader {
            audio_id,
            num_bytes: TONIE_LENGTH_MAX as u64,
            track_page_nums: vec![0u32; TONIEFILE_MAX_CHAPTERS],
            ..Default::default()
        };
        let mut toniefile = Toniefile {
            path: path.to_string(),
            file: File::create(path).expect("Could not create file"),
            file_position: 0,
            audio_length: 0,
            opus_encoder: OpusEnc::new(
                SampleRate::Hz48000,
                audiopus::Channels::Stereo,
                audiopus::Application::Audio,
            )
            .expect("Could not create opus encoder"),
            audio_frame: [0; (OPUS_FRAME_SIZE * OPUS_CHANNELS) as usize],
            audio_frame_used: 0,
            ogg_stream: OggStream::new(audio_id as i32),
            ogg_granulepos: 0,
            ogg_packet_count: 0,
            header,
            taf_block_number: 0,
            n_taf_blocks: 0,
            sha1: Sha1::new(),
        };
        toniefile.new_chapter();

        toniefile.write_header();
        toniefile
            .file
            .seek(SeekFrom::Start(TONIEFILE_FRAME_SIZE as u64))
            .expect("Could not seek to frame size");

        // opus settings
        toniefile
            .opus_encoder
            .set_bitrate(Bitrate::BitsPerSecond(OPUS_BITRATE as i32))
            .expect("Could not set bitrate");
        toniefile
            .opus_encoder
            .set_vbr(false)
            .expect("Could not set vbr");
        toniefile
            .opus_encoder
            .set_encoder_ctl_request(
                ffi::OPUS_SET_EXPERT_FRAME_DURATION_REQUEST,
                OPUS_FRAME_SIZE_MS_OPUS,
            )
            .expect("Could not set frame duration");

        let opus_header: [u8; 19] = [
            b'O',
            b'p',
            b'u',
            b's',
            b'H',
            b'e',
            b'a',
            b'd',
            0x01,
            OPUS_CHANNELS as u8,
            0x38,
            0x01,
            (OPUS_SAMPLE_RATE as u8) & 0xFF,
            (OPUS_SAMPLE_RATE >> 8) as u8,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
        ];
        let mut opus_tags: Vec<u8> = Vec::with_capacity(0x1B4);
        opus_tags.extend(b"OpusTags");
        toniefile.comment_add(
            &mut opus_tags,
            &format!("toniefile {}", std::env!("CARGO_PKG_VERSION")),
        );
        // TODO add some opus user comments here

        // add padding of the first block
        let padding_str = "pad=";
        let mut len_bf = [0u8; 4];
        LittleEndian::write_u32(
            &mut len_bf,
            (opus_tags.capacity() - opus_tags.len() - 4) as u32,
        );
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
            toniefile
                .file
                .write_all(&og.header)
                .expect("Could not write ogg header");
            toniefile
                .file
                .write_all(&og.body)
                .expect("Could not write ogg body");
            toniefile.file_position += (og.header.len() + og.body.len()) as u64;
            toniefile.audio_length += (og.header.len() + og.body.len()) as u32;

            toniefile.sha1.update(&og.header);
            toniefile.sha1.update(&og.body);
        }

        Ok(toniefile)
    }
    /// Finalize the toniefile by writing the last ogg page and the sha1 hash
    /// in the header
    pub fn close(&mut self) {
        todo!();
    }

    pub fn new_chapter(&mut self) -> Result<()> {
        if self.header.track_page_nums.len() > TONIEFILE_MAX_CHAPTERS {
            return Err(anyhow!("Maximum number of chapters reached"));
        }

        self.header.track_page_nums[self.n_taf_blocks] = self.taf_block_number;
        self.n_taf_blocks += 1;
        println!("new chapter at {:x}", self.taf_block_number);

        Ok(())
    }

    pub fn encode(&mut self, sample_buf: &[i16], samples_available: usize) -> Result<()> { // TODO get rid of samples available, use sample_buf.len()
        const PAGE_HEADER_SIZE: i64 = 27;
        let mut samples_processed = 0;
        let mut output_frame = [0u8; TONIEFILE_FRAME_SIZE];

        while samples_processed < samples_available {
            let mut samples = OPUS_FRAME_SIZE as usize - self.audio_frame_used as usize;
            let samples_remaining = samples_available - samples_processed;
            if samples > samples_remaining {
                samples = samples_remaining;
            }

            // copy segment of sample_buf to audio_frame
            self.audio_frame
                [self.audio_frame_used as usize..self.audio_frame_used as usize + samples]
                .copy_from_slice(&sample_buf[samples_processed..samples_processed + samples]);
            self.audio_frame_used += samples as isize;
            samples_processed += samples;

            if self.audio_frame_used < OPUS_FRAME_SIZE as isize {
                continue;
            }
            let mut page_used = (self.file_position % OPUS_FRAME_SIZE as u64) as i64
                + PAGE_HEADER_SIZE
                + self.ogg_stream.get_lacing_fill() - self.ogg_stream.get_lacing_returned()
                + self.ogg_stream.get_body_fill() - self.ogg_stream.get_body_returned();
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
            if frame_payload < OPUS_PACKET_MINSIZE as i64 {
                eprintln!("Not enough space in this block!");
                return Err(anyhow!("Not enough space in this block!"));
            }

            println!("audio frame len {}", &self.audio_frame[..OPUS_FRAME_SIZE as usize].len());
            println!("output frame len {}", output_frame[..frame_payload as usize].len());
            let mut frame_len = self
                .opus_encoder
                .encode(&self.audio_frame[..2*OPUS_FRAME_SIZE as usize], &mut output_frame[..frame_payload as usize])
                .expect("Could not encode opus frame");
            if frame_payload - (frame_len as i64) < (OPUS_PACKET_PAD as i64) {
                let target_length = frame_payload;
                packet_pad(&mut output_frame[..], target_length as i32).expect("Could not pad opus frame");
                frame_len = target_length as usize;
            }

            /* we have to retrieve the actually encoded samples in this frame */
            let mut nb_frames = 0;
            unsafe {
                    nb_frames = ffi::opus_packet_get_nb_frames(output_frame.as_ptr(), frame_len as i32);
            }
            let frames = samples_per_frame(&output_frame[..], SampleRate::Hz48000).expect("Could not calculate frames") * nb_frames as usize;
            if frames != OPUS_FRAME_SIZE as usize {
                eprintln!("Encoded frame size mismatch! Got {}, should be {}", frames, OPUS_FRAME_SIZE);
            }
            self.ogg_granulepos += frames as u64 ;

            let mut packet = OggPacket::new(&output_frame);
            packet.set_bos(false);
            packet.set_eos(false);
            packet.set_granulepos(self.ogg_granulepos as i64);
            packet.set_packetno(self.ogg_packet_count);

            self.ogg_packet_count += 1;

            self.ogg_stream.packetin(&mut packet);

            page_used = (self.file_position % OPUS_FRAME_SIZE as u64) as i64
                + PAGE_HEADER_SIZE
                + self.ogg_stream.get_lacing_fill()
                + self.ogg_stream.get_body_fill();
            page_remain = TONIEFILE_FRAME_SIZE as i64 - page_used as i64;

            if page_remain < TONIEFILE_PAD_END as i64 {
                if page_remain > 0 {
                    eprintln!("unexpected small padding at {} ( {}  s)", self.ogg_granulepos, self.ogg_granulepos / OPUS_FRAME_SIZE as u64 * 60 / 1000);
                    return Err(anyhow!("unexpected small padding at {} ( {}  s)", self.ogg_granulepos, self.ogg_granulepos / OPUS_FRAME_SIZE as u64 * 60 / 1000));
                }

                while let Some(og) = self.ogg_stream.flush() {
                    self
                        .file
                        .write_all(&og.header)
                        .expect("Could not write ogg header");
                    self
                        .file
                        .write_all(&og.body)
                        .expect("Could not write ogg body");
                    let prev = self.file_position;
                    self.file_position += (og.header.len() + og.body.len()) as u64;
                    self.audio_length += (og.header.len() + og.body.len()) as u32;

                    self.sha1.update(&og.header);
                    self.sha1.update(&og.body);

                    if prev / TONIEFILE_FRAME_SIZE as u64 != self.file_position / TONIEFILE_FRAME_SIZE as u64 {
                        self.taf_block_number += 1;
                        if self.file_position % TONIEFILE_FRAME_SIZE as u64 != 0 {
                            eprintln!("Block alignment mismatch at 0x{:x}", self.file_position);
                            return Err(anyhow!("Block alignment mismatch at 0x{:x}", self.file_position));
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

    #[test]
    fn create_toniefile() {
        let toniefile = Toniefile::create(
            get_test_path().join("test.tonie").to_str().unwrap(),
            0x12345678,
        );
        // println!("{:x?}", toniefile);
        assert!(toniefile.is_ok());
    }

    #[test]
    fn fill_toniefile() {
        let mut toniefile = Toniefile::create(
            get_test_path().join("test.tonie").to_str().unwrap(),
            0x12345678,
        ).unwrap();
        toniefile.write_header();
        let mut samples = vec![0i16; TONIEFILE_FRAME_SIZE*16];
        // fill with garbage
        for i in 0..TONIEFILE_FRAME_SIZE*16 {
            samples[i] = i as i16;
        }
        toniefile.encode(&samples, TONIEFILE_FRAME_SIZE*16);
        assert!(true);
    }
}
