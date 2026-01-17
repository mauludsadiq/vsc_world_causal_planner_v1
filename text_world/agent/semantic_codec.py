from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class Sid16Factors:
    topic_id: int
    speech_act: int
    polarity: int
    conf_bucket: int

def pack_sid16(f: Sid16Factors) -> int:
    topic = int(f.topic_id) & 0x01FF
    act = int(f.speech_act) & 0x0007
    pol = int(f.polarity) & 0x0003
    conf = int(f.conf_bucket) & 0x0003

    sid16 = 0
    sid16 |= topic
    sid16 |= (act << 9)
    sid16 |= (pol << 12)
    sid16 |= (conf << 14)

    return int(sid16) & 0xFFFF

def unpack_sid16(sid16: int) -> Sid16Factors:
    x = int(sid16) & 0xFFFF
    topic = x & 0x01FF
    act = (x >> 9) & 0x0007
    pol = (x >> 12) & 0x0003
    conf = (x >> 14) & 0x0003
    return Sid16Factors(topic_id=int(topic), speech_act=int(act), polarity=int(pol), conf_bucket=int(conf))
