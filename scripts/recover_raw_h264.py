"""
DJI MP4 Recovery Utility (Extract H.264 Annex-B Stream from Incomplete mdat)
=============================================================================
Recovers interrupted DJI video recordings missing the moov atom header.
Converts length-prefixed AVC NAL units in mdat into Annex-B format and remuxes via FFmpeg.
"""

import os
import subprocess
import imageio_ffmpeg

INPUT_MP4 = "videos/06.MP4"
RAW_H264 = "videos/06_raw.h264"
OUTPUT_MP4 = "videos/06_recovered.mp4"


def recover_h264():
    print(f"[Recovery] Scanning {INPUT_MP4} for raw H.264 NAL units...")
    
    with open(INPUT_MP4, "rb") as fin, open(RAW_H264, "wb") as fout:
        # Seek past ftyp and free atoms to mdat payload (offset 512)
        fin.seek(512)
        
        chunk_size = 10 * 1024 * 1024 # 10MB chunk
        nal_count = 0
        
        while True:
            # Read 4-byte length header
            len_bytes = fin.read(4)
            if len(len_bytes) < 4:
                break
                
            nal_len = int.from_bytes(len_bytes, "big")
            
            # Sanity check NAL length (10 bytes to 2MB)
            if 0 < nal_len < 4 * 1024 * 1024:
                nal_payload = fin.read(nal_len)
                if len(nal_payload) < nal_len:
                    break
                
                # Write Annex-B 4-byte start code + payload
                fout.write(b"\x00\x00\x00\x01" + nal_payload)
                nal_count += 1
            else:
                # Seek forward 1 byte if out of sync
                fin.seek(-3, 1)
                
    print(f"[Recovery] Extracted {nal_count:,d} NAL units to {RAW_H264}.")
    
    # Remux Annex-B H.264 to valid MP4 container using FFmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-r", "30", # Default 30 FPS for DJI Mini 3 Pro
        "-i", RAW_H264,
        "-c:v", "copy",
        OUTPUT_MP4
    ]
    print(f"[Recovery] Remuxing to {OUTPUT_MP4} via FFmpeg...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[Recovery] SUCCESS! Recovered video saved to {OUTPUT_MP4}")
    else:
        print(f"[Recovery] FFmpeg remuxing output:\n{res.stderr}")


if __name__ == "__main__":
    recover_h264()
