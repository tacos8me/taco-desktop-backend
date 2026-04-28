# Music Video Editing — Theory + MCP Tool Reference

**Status**: v0.3 design draft. Theory + grammar are stable; tool surface ships as a single bundled v0.3.0 release across taco-backend v1.15.0 (PR 1) and noodlefinger-mcp v0.3.0 (PR 2).

**Audience**: LLM clients (Claude Code, Cursor, Codex CLI) authoring music videos via the noodlefinger-mcp server. Human reviewers debugging output. Anyone wondering why we built a composition language instead of 15 specialized tools.

**Reading paths**:
- *I'm an LLM about to write a shot list*: §5 (tool reference) → §9 (house style) → §8 (worked examples).
- *I'm a reviewer*: §2 (theory) → §3 (grammar) → §4 (algorithm) → §10 (what we can't fake).
- *I'm planning v0.4+*: §11 (roadmap) → §12 (prior art).

---

## 1. Why this doc exists

`cut_music_video()` produces N uniform clips chained by segment-uri. That's competent assembly. **It is not editing.** Editing means *intentional* cuts: cuts on the snare, audio leading the picture, palette swapping at the chorus, B-roll cadence rising into the drop. Walter Murch's "Rule of Six" calls these *rhythmic*, *eye-trace*, and *2D-plane* concerns; together they're roughly 25% of editing's cognitive load. The LLM-author owns the other 75% (emotion, story, 3D continuity).

This doc gives the LLM the 25% it can compute, plus the vocabulary to talk to the algorithm about the 75% it can't.

---

## 2. The Theory

### 2.1 Synaesthesia (Goodwin)

Andrew Goodwin (*Dancing in the Distraction Factory*, 1992) coins **synaesthesia** for the image-music coupling that defines music video. He distinguishes three modes of cut-music relationship:

- **Illustrative**: the cut shows what the lyric says ("she walks away" → cut to her walking away).
- **Amplifying**: the cut intensifies the music's energy (snare hit → smash cut to a new shot).
- **Disjunctive**: the cut deliberately fights the music (held shot through a chorus, cut on a vocal *rest*).

Most MTV grammar is **amplifying**. Disjunctive cuts are reserved for art-directed indie / new-wave / hip-hop conceptual work.

Our composition language defaults to amplifying: cuts land on beats, density rises with energy. Disjunctive cuts are expressible (set `transition_to_next="cut"` mid-bar, override the algorithm's grid) but never default.

### 2.2 Murch's Rule of Six → algorithmic split

Walter Murch (*In the Blink of an Eye*, 1995) ranks cut criteria: emotion 51% / story 23% / rhythm 10% / eye-trace 7% / 2D plane 5% / 3D space 4%. Translated to AI music video:

| Murch criterion | Weight | Owner |
|---|---|---|
| Emotion | 51% | **LLM-author** (writes the prompts, picks the shots) |
| Story | 23% | **LLM-author** (treatment, narrative shape) |
| Rhythm | 10% | **Algorithm** (beat grid, energy curve) |
| Eye-trace | 7% | **Algorithm** (per-section density priors) |
| 2D plane | 5% | **Algorithm** (cut-on-the-grid, framing variety) |
| 3D space | 4% | **LLM-author** (continuity, match-cut intent) |

The algorithm gets ~22% of Murch's weight. The LLM-author gets ~78%. The split matters: the algorithm guarantees musicality (cuts land where the song wants them) without claiming to author the video. The LLM gets a structural skeleton it can hang taste onto.

### 2.3 Atomic vs content-bound primitives

Of 15 named MTV-era editing techniques (§3), **10 are purely temporal**: they survive a "stitch existing clips with these rules" abstraction. Beat cut, smash cut, J/L-cut, split edit, dissolve, pop edit, flash frame, B-roll insert, palette swap, rhythmic cadence — all expressible as `(clip_id, in_frame, out_frame, audio_offset, opacity_ramp, lut)` over a beat grid. **Ghost cut** is composite-but-temporal: `flash + J-cut`.

The other **5 are content-bound**. Whip pan needs source whip; match cut needs motion correspondence; jump cut needs same-subject continuity; lyric anchor needs phoneme alignment; speed ramp needs source frames at higher temporal density. Our pipeline can *guide* these via prompt engineering (~70% reliability) but cannot *guarantee* them at the composition layer.

This split decides our design. Atomic primitives become composition-language fields. Content-bound techniques become prompt-engineering helpers (`match_cut_prompt_pair`) the LLM uses, with the explicit caveat that they're best-effort.

### 2.4 The composition language — five OTIO-minus primitives

We mirror an OTIO subset. The five primitives:

1. **clip**: `{source_uri | generation_spec, in, out, start_on_timeline, speed?, audio_gain?, tail_trim_frames?, palette?, framing?}`
2. **transition**: `{between: [a, b], type ∈ {cut, xfade, smash, ghost, dip_to_black, dip_to_white}, duration_sec, audio_lead_frames?}`
3. **audio_track**: `{source_uri, offset_sec, gain_db, fade_in?, fade_out?}`
4. **beat_grid**: `{bpm, beats: [t_sec], downbeats: [t_sec], onsets: [t_sec], rms_envelope: [(t, db)]}`
5. **shot_spec** (LLM-friendly layer above clip.generation_spec): `{prompt, scale, framing, camera, palette, duration_beats | duration_sec, lora?, subject_refs?[], section?, transition_to_next?, is_insert?, flash_color?}`

Five primitives cover every MTV-era editing pattern observed across the canon. Multi-track compositing, 3D camera paths, per-frame prompt schedules, audio sub-mixes — all YAGNI for v0.3.

---

## 3. The Grammar — 14 named techniques

Each row: technique / mechanical def / our primitive / canonical example.

| # | Technique | Mechanical def | Our primitive | Canonical |
|---|---|---|---|---|
| 1 | Beat cut / on-the-one | Cut within ±1 frame of a notated beat | `transition.type=cut` snapped to `beat_grid.beats[i]` | A-ha "Take On Me" (Barron, 1985) |
| 2 | Smash cut | Hard cut between maximum-contrast shots, audio attack lands on cut frame | `transition.type=smash` | Nirvana "Smells Like Teen Spirit" (Bayer, 1991) |
| 3 | J-cut | Audio of B starts 6-12 frames before picture cut to B | `transition.audio_lead_frames > 0` | Bowie "Let's Dance" (Mallet, 1983) |
| 4 | L-cut | Audio of A continues 6-12 frames past picture cut | `transition.audio_lead_frames < 0` | Whitney Houston "I Wanna Dance with Somebody" |
| 5 | Pre-lap | J-cut where incoming sound is dialogue/FX/music *from* new scene | `transition.audio_lead_frames > 0` | Murch usage in *Apocalypse Now* |
| 6 | Ghost cut | 1-3 frames of B inserted into A before the real cut, B's audio leading | `transition.type=ghost` (decomposes to flash + J-cut) | Duran Duran "Wild Boys" (Mulcahy) |
| 7 | Cross-dissolve on sustain | A→B dissolve, length = sustained vocal note | `transition.type=xfade, duration_sec=0.5-2.0` | Heart "Alone" (Callner, 1987) |
| 8 | Pop edit / accent cut | Single isolated hard cut on song's biggest accent | `transition.type=cut` snapped to onset peak | Bonnie Tyler "Total Eclipse of the Heart" |
| 9 | Match cut | Shape/motion/color of A-out ≈ B-in | LLM authors via `match_cut_prompt_pair()` (best-effort) | Peter Gabriel "Sledgehammer" (Johnson) |
| 10 | Jump cut | Same subject, small displacement, violates 30° rule | `transition.type=cut` with same-subject prompts | Madonna "Vogue" (Fincher) |
| 11 | Whip pan | Camera pans >50% motion-blurred for ≥3-5 frames; cut hidden in blur | Best-effort via prompt; ffmpeg motion-mask fallback | Beastie Boys "Sabotage" (Jonze) |
| 12 | Strobe / flash frame | 1-3 frame insert of solid color or unrelated image | `shot.flash_color="#FFFFFF"` synthetic clip | Mötley Crüe "Wild Side" (Isham, 1987) |
| 13 | Insert / cutaway / B-roll | Non-A-line shot 8-24 frames inserted, returns to A-line | `shot.is_insert=true` | Robert Palmer "Addicted to Love" |
| 14 | Section grading / palette swap | Distinct LUT per section, hard switch at boundary | `shot.palette` per-section | Aerosmith "Cryin'" (Callner) |
| 15 | Speed ramp | Non-linear playback rate | `clip.speed != 1.0` | Björk "Army of Me" (Gondry) |
| 16 | Rhythmic cadence | Cut spacing halves over N bars approaching chorus | algorithm + `shot.duration_beats` per shot | Buggles "Video Killed the Radio Star" (Mulcahy) |

(Ghost cut and pre-lap are sub-cases of J-cut at the mechanics level, but exposed as distinct primitives for LLM clarity.)

---

## 4. The Algorithm — deterministic cut placement

Inputs: `beat_grid`, optional section labels, creative direction (prompt + duration + num_clips target), genre hint.

Output: ordered list of `ClipPlan` records — start/end beats, section, density, transition, energy_target, holdable flag.

```python
def plan_cuts(beat_grid, sections, direction, genre):
    section_plans = []
    for sec in sections:
        prior = GENRE_PRIORS[genre][sec.label]  # cuts/bar range
        energy = mean_rms_db(beat_grid.rms, sec.t0, sec.t1)
        e = normalize(energy, beat_grid.rms_min, beat_grid.rms_max)
        cpb = prior.cpb_min + (prior.cpb_max - prior.cpb_min) * (e ** 2)
        section_plans.append((sec, cpb, prior))

    candidates = []
    for sec, cpb, prior in section_plans:
        bars = bars_between(beat_grid, sec.t0, sec.t1)
        n_cuts = round(bars * cpb)
        cuts = distribute_metric(beat_grid, sec, n_cuts, prefer="downbeat")
        for c in cuts:
            candidates.append(Candidate(beat=c, section=sec, source="metric"))

    # Onset-aware snap: chorus and drops snap to transient if strong
    for c in candidates:
        nearby = onsets_within(beat_grid.onsets, c.t_sec, window=±0.5*beat_dur)
        strongest = max(nearby, key=onset_strength, default=None)
        if strongest and onset_strength(strongest) > ACCENT_THRESHOLD \
           and is_accent_section(c.section):
            c.t_sec = strongest.t
            c.source = "accent"

    # Half-time ramp into chorus
    for sec in sections:
        if sec.label == "chorus":
            apply_halving_ramp(candidates, before=sec.t0, bars=4)
            insert_smash_cut(candidates, at=sec.t0)
            extend_hold(candidates, after=sec.t0, bars=1)

    # Match-cut budget: 5-15% of cuts marked as holdable
    budget = round(len(candidates) * MATCH_CUT_RATIO[genre])
    candidates = mark_holdable(candidates, budget)

    return [
        ClipPlan(idx=i, start_beat=a.beat, end_beat=b.beat,
                 start_t_sec=a.t_sec, end_t_sec=b.t_sec,
                 section=a.section.label,
                 density=density_label(b.beat - a.beat),
                 transition=pick_transition(a, b, prior),
                 energy_target=energy_at(beat_grid, a.t_sec, b.t_sec),
                 holdable=a.holdable, snap_source=a.source,
                 bar_phase=a.beat % 16 // 4)
        for i, (a, b) in enumerate(pairs(candidates))
    ]
```

### 4.1 Per-section density priors

| Section | cuts/bar | shot bars | Notes |
|---|---|---|---|
| Intro | 0.25 | 4 | Atmosphere, often 1 shot per 4-bar phrase |
| Verse | 0.5 | 2 | Steady cadence |
| Pre-chorus | 1.0 → 2.0 | 1 → 0.5 | Halving ramp 4→2→1→0.5 bars |
| Chorus | 2.0–4.0 | 0.5 (2 beats) | Peak density, accent cuts on snare |
| Bridge | 0.125 OR 4.0 | 8 OR 0.25 | Bimodal: long hold OR rapid montage |
| Outro | 0.125 | 8+ | Dissolve, usually one final long shot |

ASL (average shot length) for chart pop sits at 1.5-2.5 s, ~2× film-narrative density (Cutting et al. 2010).

### 4.2 Onset vs grid resolution

- **Chorus / drop / pre-chorus tail**: snap to onset within ±0.5 beat if strength > μ + 1.5σ (accent override, Murch's "physiological cut").
- **Verse / intro / outro**: honor the metric grid, ignore onsets unless they ARE the downbeat.
- **Bridge**: editor's choice; algorithm decides on energy delta.

### 4.3 Energy → density (quadratic)

`cpb = cpb_min + (cpb_max - cpb_min) * (rms_norm ** 2)`. Quadratic gives verses a calm baseline and reserves high cut density for actual energy peaks. Stepwise-per-section is the v1 fallback when RMS is noisy.

### 4.4 Genre priors (8 supported)

| Genre | Verse cpb | Chorus cpb | Match-cut budget | Quirks |
|---|---|---|---|---|
| Power ballad | 0.5 | 1.0 | 15% | 4-bar verse, sustain on title hook |
| Hair metal | 1.5 | 2.5 | 5% | 1-beat chorus, B-roll on hands |
| New wave | 1.0 | 1.5 | 5% | Rigid grid, no rubato |
| Synth-pop | 1.0 | 1.75 | 8% | Cut on every kick |
| 90s hip-hop | 2.0 | 2.75 | 20% | Lyric-anchored, fisheye |
| Indie | 0.75 | 1.0 | 20% | 4-8 bar holds, sparing jump-cuts |
| Modern pop | 1.5 | 2.5 | 10% | Per-section palette swap, dance-break |
| Country | 0.5 | 1.0 | 15% | Wide land, slow push |

Full genre presets in `flows.json` under `plan-shot-list-by-genre`.

### 4.5 Match-cut budget

Reserve 5-20% of cuts as **holdable** — the LLM-author may extend any holdable clip by +1 bar to land a match cut. Lower-energy cuts preferred for holding (chorus cuts are load-bearing rhythmically). Budget enforced globally so the LLM can't burn all holds in one section.

---

## 5. The Feature Surface — MCP tool reference

### 5.1 `cut_music_video(...)` — extended `shot_list[]` schema

Per-shot fields the orchestrator now accepts:

| Field | Type | Default | What it does |
|---|---|---|---|
| `prompt` | str | required | Per-shot prompt (overrides global `prompt`) |
| `duration_s` | float | from algorithm | Shot duration; snap-to-beat happens server-side |
| `audioStart_s` | float | cumsum | Where in the song this shot's audio window begins |
| `framing` | str ≤200ch | | Free-form camera/scale hint |
| `scale` | enum (ECU/CU/MCU/MS/MWS/WS/ELS) | | Standardized shot scale |
| `camera` | str | | Movement: push/pull/dolly/track/pan/whip/locked-off/handheld |
| `subject` | str | from SUBJECT LOCK | Reused verbatim across shots for continuity |
| `section` | enum (intro/verse/prechorus/chorus/bridge/outro) | | Tag for palette + algorithm |
| `palette` | str | | Color/lighting suffix appended to prompt |
| `transition_to_next` | enum (cut/xfade/smash/ghost/dip_to_black/dip_to_white) | cut | Per-boundary transition |
| `transition_duration_s` | float | type-default | Override default duration |
| `audio_lead_frames` | int | 0 | J-cut: audio of THIS shot leads its video N frames. Negative = L-cut. |
| `speed` | float | 1.0 | Per-clip playback speed (0.5 = slow-mo, 2.0 = double) |
| `is_insert` | bool | false | B-roll cutaway (no segment-chain, narrative continues) |
| `flash_color` | "#RRGGBB" | null | Synthetic flash frame, no LTX generation |

### 5.2 `get_beat_grid(audio_uri, analyzer="librosa")`

Wraps `POST /v1/music/analyze` (taco-backend). Returns `{bpm, beats[], downbeats[], onsets[], rms_envelope[]}`.

- `analyzer="librosa"` (default) — in-process, ~88% accuracy on pop music. Byte-identical to v1.15.x behavior.
- `analyzer="madmom"` (v1.16.0) — proxied to the madmom CPU sidecar on port 8095. Better downbeat accuracy (~+8% on cross-genre pop), BSD-licensed. Requires `LOAD_MADMOM=1` and the sidecar to be running; returns `503` otherwise (no silent fallback).

### 5.3 `plan_shot_list(audio_summary, prompt, genre, num_beats_per_shot=8, sections=[])`

Pure helper, no HTTP. Takes a beat grid + creative direction + genre, runs §4's algorithm, emits a fully-populated `shot_list[]`. The LLM accepts verbatim, hand-edits, or rejects. Genre presets in §4.4.

### 5.4 `weave_inserts(session_id, insert_specs)`

Adds B-roll inserts to a rendered MV at specified beat positions. Re-exports composition without re-rendering primary clips. `insert_specs = [{beat: int, prompt: str, duration_beats: int}]`.

### 5.5 `match_cut_prompt_pair(prev_prompt, framing_hint)`

Returns `(tail_prompt, head_prompt)` designed to share salient features. Heuristic prompt engineering — best-effort, ~70% reliability per LTX steerability empirical.

### 5.6 `apply_section_palette(shot_list, palette_map)`

Annotates each shot with palette/styling prompts. Pure dict merge.

### 5.7 Backend composition primitives

The composition POST body (`POST /v2/compositions`) now accepts:

- `clip.speed: float` (default 1.0) — per-clip playback speed
- `transition.audioLeadFrames: int` (default 0) — J/L cut audio offset
- `clip.historyId` accepts synthetic-flash history IDs from solid-color uploads

All other clip/transition fields unchanged (round-trip-safe).

---

## 6. The `shot_list` schema (canonical JSON)

```json
{
  "treatment": {
    "vibe": "1987 Sunset Strip, magenta neon, dry-ice smoke, chrome",
    "anchor_ref": "Mötley Crüe 'Wild Side' — hard cuts on snare",
    "subject_lock": "Lead — long black hair, white tank, silver cross, leather pants",
    "cadence": "5 shots, 2 bars each at 120 BPM = 4s, hard cuts on the 1"
  },
  "shots": [
    {
      "shot_n": 1,
      "duration_s": 4.0,
      "scale": "WS",
      "framing": "group",
      "camera": "locked-off",
      "subject": "band on smoke-filled stage, magenta backlight",
      "section": "intro",
      "palette": "low-key + neon backlit",
      "transition_to_next": "cut",
      "prompt": "wide stage shot, four-piece band silhouetted in magenta haze, dry ice rolling"
    }
  ]
}
```

LLM authors the treatment block first, then derives shots. The MCP server flattens shots into `cut_music_video(shot_list=[...])` per §5.1.

---

## 7. The `beat_grid` schema

Output of `POST /v1/music/analyze`:

```json
{
  "bpm": 124.5,
  "beats": [0.482, 0.964, 1.446, 1.928, ...],
  "downbeats": [0.482, 2.410, 4.338, ...],
  "onsets": [0.482, 0.840, 0.964, 1.213, ...],
  "rms_envelope": [[0.0, -28.4], [0.512, -22.1], ...],
  "duration_s": 156.3,
  "confidence": 0.87
}
```

`beats` = librosa beat track; `downbeats` = first beat of every bar (stride 4 by default, configurable for compound meters); `onsets` = transient peaks; `rms_envelope` = (t_sec, dBFS) pairs at 512-hop. v0.4+ adds `sections[]` from allin1 and `lyric_timestamps[]` from whisperX.

---

## 8. Worked Examples

### 8.1 Power ballad (Heart "Alone" style, 24-second cut)

```
TREATMENT
VIBE: stadium silhouettes, smoke, longing.
ANCHOR REF: Heart "Alone" (Callner 1987) — slow push, dissolve on title hook.
SUBJECT LOCK: Lead — auburn hair, white satin blouse, single tear.
CADENCE: 6 shots × 4s. Verse (0-12s) = 2-bar shots. Chorus (12-24s) = 1-bar shots + dissolve on title.

SHOTS (auto-emitted by plan_shot_list)
1 | 4.0s | MS  | single | locked-off  | Lead [LOCK] at piano, dim room    | desat teal      | cut
2 | 4.0s | CU  | single | slow push   | Lead [LOCK] mouths verse lyric    | desat teal      | cut
3 | 4.0s | WS  | single | crane up    | Lead [LOCK] alone on stadium stage| desat teal→amber| smash (chorus drop)
4 | 2.0s | CU  | single | slow push   | Lead [LOCK] head tilted back     | warm amber      | cut
5 | 2.0s | WS  | single | locked-off  | wide stadium reveal, lighter wave | warm amber      | xfade duration_sec=2.0 (sustain on "alone")
6 | 8.0s | ECU | single | locked-off  | tear rolling, slow                 | warm amber      | (final)
```

### 8.2 Hair metal (Mötley Crüe "Wild Side" style, 20-second cut)

```
TREATMENT
VIBE: 1987 Sunset Strip, magenta neon, dry-ice smoke.
ANCHOR REF: Mötley Crüe "Wild Side" (Isham 1987) — hard cuts on snare.
SUBJECT LOCK: Lead — long black hair, white tank, silver cross, leather pants.
CADENCE: 12 shots × ~1.6s. Verse 1-2 bars. Chorus 1-beat + 2 strobe inserts.

SHOTS (excerpt — flash inserts on snare 1 and snare 3 of chorus bar 1)
1  | 4.0s | WS  | group   | locked-off    | band on smoke-filled stage          | low-key + neon | cut
2  | 4.0s | ECU | insert  | locked-off    | fingers on fretboard mid-bend       | hard top       | cut
3  | 4.0s | MCU | single  | slow push     | Lead [LOCK] mouths chorus lyric     | rim + smoky    | smash (chorus)
4  | 0.083s | flash | -    | -             | -                                   | flash_color="#FFFFFF" | cut
5  | 2.0s | CU  | single  | whip pan L    | Lead [LOCK] at mic                  | low-key        | cut
6  | 0.083s | flash | -    | -             | -                                   | flash_color="#FFFFFF" | cut
7  | 2.0s | MWS | two-shot| handheld drift| Lead + bassist back-to-back        | neon + smoky   | cut
... (8 more shots through outro)
```

### 8.3 90s hip-hop (Hype Williams style, 16-second cut)

```
TREATMENT
VIBE: mirrored chrome set, all-red wash, fisheye flex.
ANCHOR REF: Missy Elliott "The Rain" (Hype Williams 1997) — fisheye CU + saturated single-color.
SUBJECT LOCK: MC — chrome jumpsuit, oversized shades, gold chain.
CADENCE: 8 shots × 2s. Lyric-anchored — cut on the "1" of every bar.

SHOTS
1 | 2.0s | WS  | single   | locked-off | MC [LOCK] in all-red mirror room          | hyper-sat red | cut
2 | 2.0s | CU  | single   | fisheye    | MC [LOCK] mouths punchline (fisheye distort) | sat red    | cut
3 | 2.0s | MS  | insert   | locked-off | gold chain swinging, chrome reflection      | sat red    | smash
4 | 2.0s | CU  | single   | low-angle  | MC [LOCK] hero pose, chrome ceiling reflects | sat gold  | cut
5 | 2.0s | WS  | group    | dolly arc  | MC + 4 dancers, all-gold set                | sat gold   | cut
6 | 2.0s | CU  | single   | fisheye    | MC [LOCK] direct-to-camera flex             | sat gold   | strobe (3-frame)
7 | 2.0s | MS  | two-shot | tracking   | MC + featured artist, mirror-floor          | sat gold   | cut
8 | 2.0s | ELS | single   | crane up   | MC [LOCK] center of room, reveal scale       | sat gold   | (final)
```

---

## 9. The LLM House Style

Embedded in `get_flow("cut-music-video")`. The LLM follows this template:

1. **Treatment first, ~120 words**: VIBE / ANCHOR REF / SUBJECT LOCK / CADENCE.
2. **Shot list, fixed schema**: shot_n, duration_s, scale, framing, camera, subject (reuses LOCK verbatim), palette, transition_to_next, prompt (≤30 words collapsing the rest).
3. **Validation pass**:
   - Subject phrase appears verbatim in every performance shot.
   - No prompt > 30 words.
   - No two adjacent shots share scale (avoid CU→CU stutter unless explicit match cut).
   - Camera-movement count ≤ 60% of shots.
   - Lists ≥ 8 shots include ≥ 1 ECU and ≥ 1 WS.
   - No "cinematic", "epic", "stunning", "8k", "masterpiece", "award-winning".

### 9.1 Shot vocabulary (controlled terms)

- **Scale**: ECU, CU, MCU, MS, MWS, WS, ELS
- **Framing**: single, two-shot, group, OTS, POV, insert
- **Camera**: push, pull, dolly-L/R, track, pan, tilt, whip, crane, handheld, locked-off, orbit
- **Lighting**: high-key, low-key, motivated, practical, neon, smoky, backlit, rim, hard top, bounce
- **Performance verbs**: mouths-lyric, head-bangs, struts, locks-eyes-camera, holds-mic, shreds-solo

### 9.2 Anchor references (named MVs by genre)

| Genre | Anchor | Cut grammar to imitate |
|---|---|---|
| Hair metal | Mötley Crüe "Wild Side" (1987) | Hard cuts on snare, smoky low-key + magenta backlight |
| Power ballad | Heart "Alone" (1987) | Slow push, dissolve on title hook |
| Surreal pop | Björk "It's Oh So Quiet" (Jonze 1995) | Locked-off WS verse, choreographed crane chorus |
| Hip-hop | Missy Elliott "The Rain" (Williams 1997) | Fisheye CU, saturated single-color sets, whip-pan |
| Indie | Weezer "Buddy Holly" (Gondry 1994) | Period-pastiche locked-off MWS, minimal movement |
| Modern pop | Dua Lipa "Houdini" (2023) | One-take handheld orbit, rim light |

### 9.3 Failure modes the LLM should avoid

| Failure | Symptom | Fix |
|---|---|---|
| Subject drift | Singer's hair changes every shot | SUBJECT LOCK reused verbatim |
| Genre drift | Hair-metal brief → A24-prestige output | Mandatory anchor-MV citation |
| Cadence drift | 8 equal shots regardless of song | Pre-commit shot count + beats |
| Adjective inflation | "epic cinematic stunning" stack | Negative-prompt list |
| Movement saturation | Every shot moves; output feels seasick | Cap moving shots at 60% |
| Scale monotony | All MS performance | Require ≥1 ECU + ≥1 WS |

---

## 10. What you can't fake

LTX-2.3 honors content-bound prompts at ~70% reliability. Whip pans don't always whip. Match cuts don't always match. Solid-white frames have residual noise. We do NOT promise these as primitives. We promise them as *preferences* the LLM expresses via prompt engineering. Specifically:

- **Whip pan**: ~70% prompt-honoring. Fallback: ffmpeg `motion_blur` post-pass on the seam. v0.4+ feature.
- **Match cut**: ~70% via paired prompts. v0.5+: visual frame-embedding match-cut detection.
- **Solid color frame**: synthetic upload (no LTX generation). Already supported via `flash_color` field.
- **Lip-sync**: needs phoneme alignment. v0.4+ via whisperX post-pass over ACE-Step output.
- **Speed ramp** *within* a clip (variable speed): v0.5+; uniform per-clip speed shipped in v0.3.0 PR 1.

Be honest about the ceiling. The LLM that reads this doc should know what to fall back to.

---

## 11. Roadmap (v0.4+)

### ✅ Shipped in v1.16.0

- **madmom downbeat sidecar** — better bar-1 cuts (~+8% accent-cut accuracy on cross-genre pop). CPU-only FastAPI service on port 8095, BSD-licensed. Opt-in via `analyzer="madmom"` on `POST /v1/music/analyze`. See backend `CLAUDE.md` → "madmom downbeat sidecar (v1.16.0)" for setup + ops.

### ✅ Shipped in noodlefinger-mcp v0.4.2

- **Per-shot audio slicing** — the `cut_music_video` orchestrator now pre-slices the song into per-clip audio windows using the `shot_list[].start_s` / `end_s` schedule, so each underlying a2v call's motion conditions on the *correct* audio segment for its shot rather than on the full song. Closes a long-standing artifact where motion drifted toward the prevailing energy of the whole track instead of the bar-local pulse. Lives entirely in the MCP layer; backend `/v2/audio-to-video` is unchanged.

### Deferred

- ~~whisperX lyric alignment~~ — lyric-anchor cuts at word granularity. **Deferred** to a future major: ACE-side lyric timestamps were investigated as an alternative (whisperX post-pass on the generated track) and judged not worth the operational complexity at current scale.
- ~~allin1 section detection~~ — automatic verse/chorus/bridge labels. **Deferred**: blocked on CC-BY-NC weights (allin1's pretrained checkpoints are non-commercial-only). Re-evaluate if the model is re-licensed or a permissive equivalent ships.

### Pending

- visual match-cut detection (frame-embedding search) — closes the §10 match-cut gap
- per-clip ffmpeg LUT color grading — palette swap as composition primitive, not just prompt suffix
- OTIO/FCPXML export alongside MP4 — pro NLE handoff for finishing in Premiere/Resolve
- Stem separation (Demucs) → drums drive cuts, vocals drive shot semantics, bass drives camera (per RyanOnTheInside ComfyUI node taxonomy)

---

## 12. Related (prior art, what we stole, what we avoided)

**Stole**: `mugen`'s per-section pacing maps onto our `audioDurationSec` + `tailTrimFrames`. RyanOnTheInside's stem-separated audio feature taxonomy informs v0.4+ Demucs work. OpenMontage's 3-layer architecture (skills/tools/pipelines) maps to our taco-backend (tools) + MCP (pipelines+skills). OpenTimelineIO's primitive set (clip/transition/track/marker) is our composition language reference.

**Avoided**: latent-walk visualizers (look 2022). Random beat-cuts on stock footage without per-section intent (mugen's failure mode). Full-LLM-as-editor (non-reproducible). Coupling editor to a single video model. Custom audio analysis (use librosa + Demucs vocab). Flat-MP4-only output (every dead repo's mistake).

**The gap we fill**: deterministic shot-list + AI generation as separable layers. Closest published academic attempt is StoryFlow (arxiv 2505.12237). No open-source LTX-multi-clip-MV demo exists as of v0.3 design. We are first-mover in this slot.

**See also**: `docs/MCP.md` §4.5 (feature walkthrough), `docs/API.md` (backend endpoint shapes), `flows.json` (LLM-readable per-flow guides with embedded few-shot examples).
