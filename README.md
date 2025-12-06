
---

## Requirements

### Hardware
- **KrakenSDR (5-channel coherent SDR)**
- **Reference antenna**: directed toward a strong local FM or TV transmitter  
- **Surveillance antennas (4)**: spaced and oriented toward the monitored airspace  
- **Stable clock and calibrated cables** (phase coherence required)

### Software Dependencies
- [GNU Radio 3.10+](https://wiki.gnuradio.org/)
- [gr-osmosdr](https://osmocom.org/projects/gr-osmosdr/wiki) (with SoapySDR support)
- `numpy`, `PyQt5`, `sip`
- Optional: `SoapySDR` driver for KrakenRF

Install dependencies (Ubuntu/Debian):
```bash
sudo apt install gnuradio gr-osmosdr python3-pyqt5 python3-sip
pip install numpy

## System Architecture & Signal Flow

The diagram below illustrates a KrakenSDR-based passive radar with **five coherent channels**: one **Reference** channel aimed at the illuminator (FM/TV) and **four Surveillance** channels covering the airspace. Each surveillance branch performs an FFT-domain **Cross-Ambiguity Function (CAF)** with the common Reference FFT, followed by IFFT and magnitude-squared to form a **range profile**; successive profiles form a **range–time** display.

<!-- Inline SVG (GitHub renders this directly) -->
<p align="center">
<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="560" viewBox="0 0 1100 560" role="img" aria-labelledby="title desc">
  <title id="title">Passive Radar with KrakenSDR: Reference + 4 Surveillance Channels</title>
  <desc id="desc">Block diagram showing Reference chain, four Surveillance chains, CAF via FFT×conj, IFFT, magnitude squared, and range–time raster outputs.</desc>
  <defs>
    <style>
      .blk{fill:#f7f9fc;stroke:#2b4c7e;stroke-width:1.6;rx:8;ry:8}
      .grp{fill:#eef3ff;stroke:#8aa4d6;stroke-width:1.2;rx:10;ry:10}
      .txt{font: 12px sans-serif; fill:#0f172a}
      .hd {font: 700 14px/1.2 sans-serif}
      .arrow{stroke:#334155;stroke-width:1.6;marker-end:url(#marr)}
      .small{font: 11px sans-serif; fill:#334155}
      .note{font: 11px sans-serif; fill:#475569}
    </style>
    <marker id="marr" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto">
      <path d="M0,0 L10,4 L0,8 z" fill="#334155"/>
    </marker>
  </defs>

  <!-- KrakenSDR group -->
  <rect x="20" y="20" width="220" height="520" class="grp"/>
  <text x="30" y="42" class="hd">KrakenSDR (5-ch coherent)</text>
  <text x="30" y="60" class="small">Common LO/clock; phase-calibrated</text>

  <!-- Channels inside Kraken group -->
  <rect x="36" y="90"  width="188" height="48" class="blk"/>
  <text x="44" y="110" class="txt">Ref ch (index 4)</text>
  <text x="44" y="125" class="small">Antenna → Illuminator</text>

  <rect x="36" y="160" width="188" height="48" class="blk"/>
  <text x="44" y="180" class="txt">Surv ch 0</text>
  <text x="44" y="195" class="small">Antenna → Airspace sector A</text>

  <rect x="36" y="230" width="188" height="48" class="blk"/>
  <text x="44" y="250" class="txt">Surv ch 1</text>

  <rect x="36" y="300" width="188" height="48" class="blk"/>
  <text x="44" y="320" class="txt">Surv ch 2</text>

  <rect x="36" y="370" width="188" height="48" class="blk"/>
  <text x="44" y="390" class="txt">Surv ch 3</text>

  <!-- Reference processing chain -->
  <rect x="290" y="90" width="160" height="44" class="blk"/>
  <text x="300" y="108" class="txt">Freq-Xlating FIR (BW)</text>
  <text x="300" y="123" class="small">channelization</text>

  <rect x="470" y="90" width="120" height="44" class="blk"/>
  <text x="480" y="108" class="txt">DC Blocker</text>

  <rect x="610" y="90" width="120" height="44" class="blk"/>
  <text x="620" y="108" class="txt">S→V + Window</text>

  <rect x="750" y="90" width="120" height="44" class="blk"/>
  <text x="760" y="108" class="txt">FFT (Ref)</text>

  <!-- Arrows ref chain -->
  <line x1="224" y1="114" x2="290" y2="112" class="arrow"/>
  <line x1="450" y1="112" x2="470" y2="112" class="arrow"/>
  <line x1="590" y1="112" x2="610" y2="112" class="arrow"/>
  <line x1="730" y1="112" x2="750" y2="112" class="arrow"/>

  <!-- Tap lines from Ref FFT to mixers -->
  <text x="760" y="82" class="small">Broadcast spectrum slice</text>
  <line x1="810" y1="134" x2="860" y2="200" class="arrow"/>
  <line x1="810" y1="134" x2="860" y2="270" class="arrow"/>
  <line x1="810" y1="134" x2="860" y2="340" class="arrow"/>
  <line x1="810" y1="134" x2="860" y2="410" class="arrow"/>

  <!-- Surveillance 0 chain -->
  <rect x="290" y="160" width="160" height="44" class="blk"/>
  <text x="300" y="178" class="txt">Freq-Xlating FIR (BW)</text>
  <rect x="470" y="160" width="120" height="44" class="blk"/>
  <text x="480" y="178" class="txt">DC Blocker</text>
  <rect x="610" y="160" width="120" height="44" class="blk"/>
  <text x="620" y="178" class="txt">S→V + Window</text>
  <rect x="750" y="160" width="120" height="44" class="blk"/>
  <text x="760" y="178" class="txt">FFT (Surv0)</text>
  <line x1="224" y1="184" x2="290" y2="182" class="arrow"/>
  <line x1="450" y1="182" x2="470" y2="182" class="arrow"/>
  <line x1="590" y1="182" x2="610" y2="182" class="arrow"/>
  <line x1="730" y1="182" x2="750" y2="182" class="arrow"/>

  <rect x="880" y="184" width="160" height="44" class="blk"/>
  <text x="890" y="202" class="txt">× conj( )</text>
  <text x="890" y="217" class="small">CAF in FFT domain</text>
  <line x1="870" y1="182" x2="880" y2="206" class="arrow"/> <!-- from Ref FFT -->
  <line x1="870" y1="182" x2="870" y2="182" style="opacity:0"/> <!-- keep for layout -->

  <line x1="870" y1="182" x2="880" y2="206" class="arrow" style="opacity:0"/>

  <!-- Surveillance 1 chain -->
  <rect x="290" y="230" width="160" height="44" class="blk"/>
  <text x="300" y="248" class="txt">Freq-Xlating FIR (BW)</text>
  <rect x="470" y="230" width="120" height="44" class="blk"/>
  <text x="480" y="248" class="txt">DC Blocker</text>
  <rect x="610" y="230" width="120" height="44" class="blk"/>
  <text x="620" y="248" class="txt">S→V + Window</text>
  <rect x="750" y="230" width="120" height="44" class="blk"/>
  <text x="760" y="248" class="txt">FFT (Surv1)</text>
  <line x1="224" y1="254" x2="290" y2="252" class="arrow"/>
  <line x1="450" y1="252" x2="470" y2="252" class="arrow"/>
  <line x1="590" y1="252" x2="610" y2="252" class="arrow"/>
  <line x1="730" y1="252" x2="750" y2="252" class="arrow"/>

  <rect x="880" y="254" width="160" height="44" class="blk"/>
  <text x="890" y="272" class="txt">× conj( )</text>
  <text x="890" y="287" class="small">CAF in FFT domain</text>

  <!-- Surveillance 2 chain -->
  <rect x="290" y="300" width="160" height="44" class="blk"/>
  <text x="300" y="318" class="txt">Freq-Xlating FIR (BW)</text>
  <rect x="470" y="300" width="120" height="44" class="blk"/>
  <text x="480" y="318" class="txt">DC Blocker</text>
  <rect x="610" y="300" width="120" height="44" class="blk"/>
  <text x="620" y="318" class="txt">S→V + Window</text>
  <rect x="750" y="300" width="120" height="44" class="blk"/>
  <text x="760" y="318" class="txt">FFT (Surv2)</text>
  <line x1="224" y1="324" x2="290" y2="322" class="arrow"/>
  <line x1="450" y1="322" x2="470" y2="322" class="arrow"/>
  <line x1="590" y1="322" x2="610" y2="322" class="arrow"/>
  <line x1="730" y1="322" x2="750" y2="322" class="arrow"/>

  <rect x="880" y="324" width="160" height="44" class="blk"/>
  <text x="890" y="342" class="txt">× conj( )</text>
  <text x="890" y="357" class="small">CAF in FFT domain</text>

  <!-- Surveillance 3 chain -->
  <rect x="290" y="370" width="160" height="44" class="blk"/>
  <text x="300" y="388" class="txt">Freq-Xlating FIR (BW)</text>
  <rect x="470" y="370" width="120" height="44" class="blk"/>
  <text x="480" y="388" class="txt">DC Blocker</text>
  <rect x="610" y="370" width="120" height="44" class="blk"/>
  <text x="620" y="388" class="txt">S→V + Window</text>
  <rect x="750" y="370" width="120" height="44" class="blk"/>
  <text x="760" y="388" class="txt">FFT (Surv3)</text>
  <line x1="224" y1="394" x2="290" y2="392" class="arrow"/>
  <line x1="450" y1="392" x2="470" y2="392" class="arrow"/>
  <line x1="590" y1="392" x2="610" y2="392" class="arrow"/>
  <line x1="730" y1="392" x2="750" y2="392" class="arrow"/>

  <rect x="880" y="394" width="160" height="44" class="blk"/>
  <text x="890" y="412" class="txt">× conj( )</text>
  <text x="890" y="427" class="small">CAF in FFT domain</text>

  <!-- IFFT, |·|², and Raster per surveillance -->
  <!-- Surv0 outputs -->
  <rect x="880" y="214" width="160" height="44" class="blk" transform="translate(0,40)"/>
  <text x="900" y="272" class="txt">IFFT</text>
  <line x1="960" y1="228" x2="960" y2="254" class="arrow"/> <!-- down from mixer -->
  <line x1="960" y1="294" x2="960" y2="328" class="arrow"/>
  <rect x="880" y="328" width="160" height="44" class="blk"/>
  <text x="892" y="346" class="txt">|·|² (Range profile)</text>
  <line x1="960" y1="372" x2="1060" y2="372" class="arrow"/>
  <text x="980" y="365" class="small">frames →</text>
  <rect x="1060" y="352" width="28" height="40" class="blk"/>
  <text x="1092" y="375" class="small">Raster</text>

  <!-- Surv1/2/3 notes -->
  <text x="1040" y="280" class="note">Repeat per surveillance channel (×4)</text>

  <!-- Legend -->
  <rect x="290" y="460" width="750" height="70" class="grp"/>
  <text x="300" y="480" class="hd">Legend & Notes</text>
  <text x="300" y="498" class="note">BW: choose ~180 kHz for FM (or a narrow TV pilot/sub-band). Window: Hann/Hamming. CAF: FFT(ref) × conj(FFT(surv)).</text>
  <text x="300" y="514" class="note">Successive range profiles form a range–time raster; motion appears as sloped traces. Add slow-time FFT per bin for Doppler.</text>

  <!-- Inputs text -->
  <text x="28" y="82" class="small">Inputs from coherent RF front-end</text>
</svg>
</p>

### Diagram Highlights
- **Reference chain**: isolates the direct-path broadcast slice and produces a single FFT that fans out to all branches.  
- **Surveillance chains (×4)**: identical processing; each performs **CAF** with the Reference FFT, then **IFFT → |·|²** to produce a range profile.  
- **Range–Time**: stacking profiles across time yields the range–time raster per surveillance channel.

---

## Mermaid Fallback (Optional)

If you prefer a text-based diagram alongside the SVG (GitHub renders Mermaid), include:

```mermaid
flowchart LR
  subgraph K[KrakenSDR (5-ch coherent)]
    R[Ref ch (index 4)]
    S0[Surv ch 0]
    S1[Surv ch 1]
    S2[Surv ch 2]
    S3[Surv ch 3]
  end

  R --> F1[Freq-Xlating FIR (BW)]
  F1 --> D1[DC Blocker]
  D1 --> W1[S→V + Window]
  W1 --> FR[FFT (Ref)]

  S0 --> F0[Freq-Xlating FIR]
  F0 --> D0[DC Blocker]
  D0 --> W0[S→V + Window]
  W0 --> FS0[FFT (Surv0)]
  FR --> M0[(× conj)]
  FS0 --> M0
  M0 --> I0[IFFt]

  I0 --> P0[|·|² → Range profile]
  P0 --> RT0[[Range–Time Raster]]

  S1 --> F2[Freq-Xlating FIR]
  F2 --> D2[DC Blocker]
  D2 --> W2[S→V + Window]
  W2 --> FS1[FFT (Surv1)]
  FR --> M1[(× conj)]
  FS1 --> M1
  M1 --> I1[IFFT]
  I1 --> P1[|·|² → Range profile]
  P1 --> RT1[[Range–Time Raster]]

  S2 --> F3 --> D3 --> W3 --> FS2
  FR --> M2
  FS2 --> M2
  M2 --> I2 --> P2 --> RT2

  S3 --> F4 --> D4 --> W4 --> FS3
  FR --> M3
  FS3 --> M3
  M3 --> I3 --> P3 --> RT3



Tip: For Doppler, buffer range profiles per bin across slow-time and apply an FFT to produce a range–Doppler heatmap.

