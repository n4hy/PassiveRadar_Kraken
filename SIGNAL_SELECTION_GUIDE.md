# Block B3 Signal Selection Guide

**Quick reference for choosing the right signal of opportunity**

---

## Decision Tree

```
Are you in the US?
├─ YES → Do you need long range (>10 km)?
│        ├─ YES → Use FM Radio (Phase 2) ✅ RECOMMENDED
│        │        Backup: LTE (Phase 5)
│        │
│        └─ NO → Do you need high resolution (<10 m)?
│                 ├─ YES → Use WiFi (Phase 4) or LTE (Phase 5)
│                 └─ NO → Use FM Radio (Phase 2) ✅ SIMPLEST
│
└─ NO → What region?
         ├─ Europe/Australia → Use DVB-T (Phase 3) or FM (Phase 2)
         ├─ China → Use DTMB (future) or FM (Phase 2)
         └─ Other → Use FM Radio (Phase 2) ✅ UNIVERSAL
```

---

## Signal Comparison at a Glance

### 🎯 FM Radio - BEST FOR US USERS
- **Range:** 60+ km ⭐⭐⭐⭐⭐
- **Availability (US):** Everywhere ⭐⭐⭐⭐⭐
- **Complexity:** Very Simple ⭐
- **Implementation Time:** 2 weeks ⭐⭐⭐⭐⭐
- **CPU Load:** Very Low (5%) ⭐⭐⭐⭐⭐
- **Resolution:** Moderate (15-20 m) ⭐⭐⭐
- **Use Case:** General purpose, wide-area coverage

### 📱 LTE - BEST FOR URBAN
- **Range:** 5-30 km ⭐⭐⭐
- **Availability (US):** Everywhere ⭐⭐⭐⭐⭐
- **Complexity:** Very High ⭐⭐⭐⭐⭐
- **Implementation Time:** 5 weeks ⭐⭐
- **CPU Load:** High (60-80%) ⭐⭐
- **Resolution:** Good (5-10 m) ⭐⭐⭐⭐
- **Use Case:** Urban/suburban, moderate range

### 📡 WiFi - BEST FOR SHORT RANGE
- **Range:** 50-200 m ⭐
- **Availability (US):** Very High ⭐⭐⭐⭐
- **Complexity:** Moderate ⭐⭐⭐
- **Implementation Time:** 3 weeks ⭐⭐⭐
- **CPU Load:** Moderate (30-50%) ⭐⭐⭐⭐
- **Resolution:** Excellent (1-3 m) ⭐⭐⭐⭐⭐
- **Use Case:** Indoor, building security, short-range

### 📺 DVB-T - BEST FOR EUROPE
- **Range:** 40+ km ⭐⭐⭐⭐
- **Availability (US):** None ⭐
- **Complexity:** High ⭐⭐⭐⭐
- **Implementation Time:** 4 weeks ⭐⭐⭐
- **CPU Load:** Moderate-High (40-60%) ⭐⭐⭐
- **Resolution:** Good (8-12 m) ⭐⭐⭐⭐
- **Use Case:** Europe, Australia, international

### 🚀 5G NR - BEST FOR FUTURE
- **Range:** 1-10 km ⭐⭐
- **Availability (US):** Growing ⭐⭐⭐
- **Complexity:** Extreme ⭐⭐⭐⭐⭐
- **Implementation Time:** 7 weeks ⭐
- **CPU Load:** Very High (80-95%) ⭐
- **Resolution:** Excellent (1-2 m) ⭐⭐⭐⭐⭐
- **Use Case:** Future-proof, high-resolution

---

## Use Case Matrix

| Your Goal | Recommended Signal | Alternative | Why? |
|-----------|-------------------|-------------|------|
| **Wide-area surveillance (US)** | FM Radio | LTE | Long range, simple, ubiquitous |
| **Urban monitoring** | LTE | WiFi + FM | Good balance of range and resolution |
| **Building security** | WiFi | LTE | Short-range, high resolution |
| **Airport/port security** | FM Radio | LTE | Wide area coverage |
| **Border monitoring** | FM Radio | LTE | Long range needed |
| **Traffic monitoring** | LTE | FM | Moderate range, good resolution |
| **Indoor monitoring** | WiFi | None | Only viable option |
| **International deployment** | FM Radio | DVB-T | Universal availability |
| **Research/testing** | FM Radio | All | Simplest to get working |

---

## Geographic Availability

### United States 🇺🇸
1. **FM Radio** ✅ Everywhere
2. **LTE** ✅ Everywhere (AT&T, Verizon, T-Mobile)
3. **WiFi** ✅ Urban/suburban
4. **5G** ✅ Major cities
5. **ATSC 3.0** 🟡 Major cities only
6. **DVB-T** ❌ Not available

**Recommendation:** Start with FM, add LTE for urban areas

### Europe 🇪🇺
1. **FM Radio** ✅ Everywhere
2. **DVB-T/DVB-T2** ✅ Everywhere
3. **LTE** ✅ Everywhere
4. **WiFi** ✅ Urban/suburban
5. **5G** ✅ Major cities

**Recommendation:** DVB-T or FM, both excellent

### Australia 🇦🇺
1. **FM Radio** ✅ Everywhere
2. **DVB-T** ✅ Major cities
3. **LTE** ✅ Everywhere
4. **WiFi** ✅ Urban/suburban

**Recommendation:** DVB-T in cities, FM in rural areas

### China 🇨🇳
1. **FM Radio** ✅ Everywhere
2. **DTMB** ✅ Everywhere (Chinese digital TV)
3. **LTE** ✅ Everywhere
4. **5G** ✅ Widely deployed

**Recommendation:** FM or DTMB (when implemented)

---

## Performance Comparison

### Detection Range (with 50 kW transmitter)

| Signal | Ideal Range | Urban Range | Rural Range | Notes |
|--------|-------------|-------------|-------------|-------|
| FM Radio | 80 km | 40 km | 60 km | Best for long range |
| LTE | 30 km | 15 km | 25 km | Cell size limited |
| WiFi | 200 m | 100 m | 150 m | Very short range |
| DVB-T | 60 km | 30 km | 50 km | Similar to FM |
| 5G (sub-6) | 10 km | 5 km | 8 km | Smaller cells |

### Range Resolution

| Signal | Bandwidth | Range Resolution | Notes |
|--------|-----------|------------------|-------|
| FM Radio | ~150 kHz | 1000 m | Poor resolution |
| LTE | 20 MHz | 7.5 m | Good resolution |
| WiFi | 20-80 MHz | 1.9-7.5 m | Excellent |
| DVB-T | 8 MHz | 18.75 m | Moderate |
| 5G | 100 MHz | 1.5 m | Excellent |

**Formula:** Range resolution = c / (2 × Bandwidth)

### Doppler Resolution

Depends on coherent processing interval (CPI), not signal type.
- Typical CPI: 1 second
- Doppler resolution: ~1 Hz (all signals)
- Velocity resolution: ~0.15 m/s @ 1 GHz

---

## Implementation Complexity

### Lines of Code Estimate

| Signal | Core Algorithm | FEC/Decoding | Total | Relative |
|--------|----------------|--------------|-------|----------|
| FM Radio | 200 | 0 | 200 | 1× ⭐ SIMPLEST |
| DVB-T | 400 | 600 | 1000 | 5× |
| WiFi | 350 | 400 | 750 | 3.75× |
| LTE | 600 | 800 | 1400 | 7× |
| 5G NR | 800 | 1000 | 1800 | 9× ⭐ MOST COMPLEX |

### Dependencies

| Signal | GNU Radio Modules | External Libraries | Difficulty |
|--------|-------------------|-------------------|------------|
| FM | gr-analog | None | Easy ✅ |
| DVB-T | gr-dtv | None | Medium |
| WiFi | gr-ieee802-11 | May need custom build | Medium-Hard |
| LTE | gr-lte or custom | srsRAN (optional) | Hard |
| 5G | Custom | srsRAN or custom | Very Hard |

---

## When to Use Each Signal

### Use FM Radio When:
- ✅ You're in the US
- ✅ You need long-range detection (>20 km)
- ✅ You want simplest implementation
- ✅ You want lowest CPU load
- ✅ You're just getting started
- ✅ You need 24/7 operation
- ✅ Range resolution >20m is acceptable

### Use LTE When:
- ✅ You need moderate range (5-30 km)
- ✅ You want better resolution than FM
- ✅ You're in an urban area
- ✅ You have CPU/GPU resources
- ✅ You want to track multiple targets
- ✅ FM is not available or weak

### Use WiFi When:
- ✅ You need short-range (<200m)
- ✅ You want excellent resolution (<3m)
- ✅ You're doing indoor monitoring
- ✅ Building/room-level tracking
- ✅ High-resolution applications

### Use DVB-T When:
- ✅ You're in Europe/Australia
- ✅ You need long range
- ✅ You want good resolution (better than FM)
- ✅ You don't mind complexity

### Use 5G When:
- ✅ You want best resolution
- ✅ You want to future-proof
- ✅ You have significant compute resources
- ✅ 5G coverage is good in your area
- ✅ You don't mind high complexity

---

## Recommendation for First Implementation

### For US Users (90% of cases):
**Start with FM Radio** 🎯

Reasons:
1. Simplest to implement (2 weeks)
2. Works everywhere in US
3. Long range (60+ km)
4. Low CPU load
5. Proven for passive radar
6. Gets your system working quickly

Then add:
- **LTE** for urban areas (better resolution)
- **WiFi** for short-range/indoor

### For European Users:
**Start with DVB-T or FM**

DVB-T advantages:
- Better resolution than FM
- Purpose-built for reconstruction
- Strong signals

FM advantages:
- Simpler implementation
- More universal

### For Research/Development:
**Start with FM Radio**

Even if you eventually want LTE/5G, start with FM to:
- Validate the system works
- Learn passive radar processing
- Build foundation
- Then add complexity

---

## Migration Path

### Suggested Roadmap for US Deployment

**Month 1-2:** FM Radio (Phase 2)
- Get system working
- Validate passive radar chain
- Collect baseline performance data

**Month 3:** WiFi (Phase 4) - if needed for short-range
- Add high-resolution capability
- Indoor/building monitoring

**Month 4-5:** LTE (Phase 5)
- Add urban capability
- Better resolution than FM
- Fills the gap between FM and WiFi

**Month 6+:** 5G NR (Phase 6) - optional
- Future-proofing
- Best resolution
- Requires significant compute

---

## Quick Start Guide

### "I just want to get started in the US"
→ **Use FM Radio (Phase 2)**
- Tune to local FM station (88-108 MHz)
- Strong signal everywhere
- 2-week implementation
- Works immediately

### "I want best performance in US"
→ **Use FM for long range, LTE for urban**
- Dual-signal system
- Automatic selection based on environment
- 7-week implementation (FM + LTE)

### "I'm in Europe"
→ **Use DVB-T**
- Purpose-built for this
- 4-week implementation
- Excellent performance

### "I need indoor monitoring"
→ **Use WiFi only**
- No other option works indoors
- 3-week implementation

---

## Summary Table

| Priority | Signal | US? | Time | Range | Resolution | Complexity |
|----------|--------|-----|------|-------|------------|------------|
| 1 | FM Radio | ✅ Yes | 2 wk | 60 km | 20 m | Low |
| 2 | WiFi | ✅ Yes | 3 wk | 200 m | 2 m | Medium |
| 3 | LTE | ✅ Yes | 5 wk | 20 km | 7 m | High |
| 4 | DVB-T | ❌ No | 4 wk | 50 km | 10 m | High |
| 5 | 5G NR | ✅ Yes | 7 wk | 10 km | 1.5 m | Very High |
| 6 | ATSC 3.0 | 🟡 Cities | 5 wk | 50 km | 10 m | Very High |

---

## Bottom Line

**For US passive radar:** FM Radio first, then add LTE/WiFi as needed.

**Implementation order:**
1. ✅ Phase 1: Block skeleton (DONE)
2. 🎯 Phase 2: FM Radio (NEXT - 2 weeks)
3. Phase 4: WiFi (optional, 3 weeks)
4. Phase 5: LTE (recommended, 5 weeks)
5. Phase 6: 5G (future, 7 weeks)

**Timeline to operational system:** ~2 weeks (FM only) to ~3 months (FM + WiFi + LTE)
