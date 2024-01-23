extern crate prost_build;

fn main() {
    prost_build::compile_protos(&["src/toniebox.pb.taf-header.proto"],
                                &["src/"]).unwrap();
}
