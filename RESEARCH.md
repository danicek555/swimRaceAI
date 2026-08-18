# Výzkumný plán: stabilní box → trenérské metriky

Cíl: z boxu (~70 % kvalita) udělat spolehlivého dodavatele cropu pro rozbor
techniky, ověřit kalibraci pixely→metry a vytěžit první metriky bez pose
modelu. Každá fáze má vstup, výstup a rozhodovací kritérium.

## Fáze A — sběr dat (běží automaticky)

Tři běhy `--retrack-after-cut` na test2, pokrývají tři podmínky:

| Běh | Záběr | Dráha | Testuje |
|-----|-------|-------|---------|
| 1 | cut 3 (10,7 s) | 6 | krátká iterace, referenční dráha |
| 2 | cut 2 (16,8 s) | 8 | blízká dráha: velký plavec, hodně pěny |
| 3 | cut 2 (16,8 s) | 3 | vzdálená dráha: málo pixelů (stress test) |

Výstup: 3 videa s overlay (žlutý surový / zelený hladký / šedý NO-POSE)
+ `*_boxes.csv` segmenty v `output/test2/`.

## Fáze B — vyhodnocení (automaticky po doběhnutí)

1. CSV statistiky per běh: podíl stavů TRACKING/PREDICTED/LOST, podíl
   no_pose, distribuce `length_m`, skoky středu mezi snímky.
2. Vizuální audit: kontaktní archy v kritických momentech (obrátka,
   podvodní fáze, sprint na hladině) — symptomy podle mapy níže.
3. Kritéria úspěchu:
   - TRACKING ≥ 85 % (blízké dráhy) / ≥ 75 % (dráha 3),
   - žádný přeskok do cizí dráhy,
   - NO-POSE pokrývá podvodní fáze a nesvítí při plavání na hladině,
   - `length_m` v hladkém plavání 1,4–2,4 m.

## Fáze C — ladění (max 2 iterace na symptom)

Jedna konstanta na iteraci (`swim/config.py`), ověření re-runem POUZE
krátkého záběru (běh 1, ~10 min):

| Symptom | Konstanta |
|---|---|
| box se třese | `BOX_CENTER_ALPHA` 0.35 → 0.25 |
| box se opožďuje | `BOX_CENTER_ALPHA` 0.35 → 0.50 |
| nabírá brázdu | `BOX_MAX_LEN_M` 2.6 → 2.4, příp. `BOX_SIZE_GATE` 1.6 → 1.4 |
| ořezává paže | `BOX_MAX_LEN_M` 2.6 → 2.8 |
| PREDICTED ujíždí | `BOX_PREDICT_MAX_SECONDS` 1.0 → 0.6 |
| falešné NO-POSE (znak) | `NO_POSE_CONTRAST_RATIO` 1.30 → 1.15 |
| NO-POSE nechytá podvodní | `NO_POSE_CONTRAST_RATIO` 1.30 → 1.45 |

## Fáze D — první metriky (bez pose modelu)

Skript nad CSV (`analysis/speed_tempo.py`):
- **rychlost v(t) v m/s**: hladký střed boxu × lokální px/m z rozteče lan,
  vyhlazená křivka přes celý záběr,
- **tempo záběrů**: FFT/autokorelace předního okraje boxu → záběry/min,
- sanity: rychlost 1,5–2,2 m/s v hladkém plavání, tempo 30–60 z/min,
  propad rychlosti u obrátky a zrychlení po odrazu viditelné v křivce.

Výstup: PNG grafy + souhrnná tabulka per délka bazénu.

## Fáze E — závěrečný report a další krok

Souhrn: co drží, co ne, doporučené konstanty. Rozhodnutí o dalším kroku:
- kalibrace sedí → pose estimation (RTMPose) na perspektivně narovnaném
  cropu, jen mimo NO-POSE zóny,
- generalizace → stejný protokol na jiném videu/bazénu (test4).

## Dělba rozhodnutí

Autonomně: běhy, statistiky, archy, ladění dle mapy, re-run krátkého záběru,
skript fáze D. Jen s odsouhlasením: commit/push, změny logiky SAM3/SAM2/lan,
mazání výstupů.

---

# Výsledky (2026-08-17, autonomní běh)

## Fáze A+B — tracking po filtru (cíl ≥85 % / ≥75 % TRACKING)

| Běh | TRACKING | PREDICTED | LOST | no_pose | length_m med/p95 |
|-----|----------|-----------|------|---------|------------------|
| dráha 8 (blízká) | **94 %** | 6 % | 0 % | 21 % | 1,27 / 2,60 |
| dráha 3 (vzdálená) | **93 %** | 7 % | 0 % | 21 % | 2,32 / 2,60 |
| dráha 6 (znak, krátký) | **78 %** | 22 % | 0 % | 32 %* | 1,58 / 2,60 |

- LOST 0 % všude — každý výpadek SAM překlenula predikce.
- p95 = 2,60 m přesně: metrický strop aktivně ořezává brázdu.
- Dráha 6 pod aspirací 85 %, ale PREDICTED kryje obrátky (fyzicky bez
  informace) — žádný přeskok dráhy, žádná ztráta identity.
- Vizuální audit: zelený box drží tělo, jediný slabý symptom je mírné
  zpoždění za sprintujícím plavcem (nelazeno — nízká priorita).

## Fáze C — ladění NO-POSE (2 iterace, vyčerpáno)

Symptom: 39 % no_pose na hladinovém znaku (falešné poplachy).
1. `NO_POSE_CONTRAST_RATIO` 1,30→1,15: **bez efektu** — kontrast je na
   vzdálených drahách hluboko pod prahem, vázající je pěnová podmínka.
2. `NO_POSE_FOAM_FRAC` 0,006→0,0025: 39 % → **32 %** (část zbytku jsou
   legitimní obrátky/podvodní fáze).

**Verdikt:** pěna+kontrast je pro znak slabý diskriminátor. Doporučení:
nahradit periodicitou pohybu (oscilace surového předního okraje, kterou už
počítáme pro tempo) — periodický pohyb ⇒ plavání na hladině ⇒ póza možná.

## Fáze D — první trenérské metriky (bez pose modelu)

| Běh | průměrná rychlost | tempo (medián) |
|-----|-------------------|----------------|
| dráha 8 | 1,50 m/s | 51,5 cyklů/min |
| dráha 3 | 1,27 m/s | 63,4 cyklů/min |
| dráha 6 | 0,75 m/s** | 63,4 cyklů/min (jen 2 okna) |

- Hodnoty fyzikálně věrohodné (po maskování švů segmentů >3 m/s).
- Křivka dráhy 8 ukazuje obrátku (propad→odraz) i intra-cyklickou variaci
  rychlosti u motýlka — reálný biomechanický jev, metrika zdarma.
- ** Krátký záběr s obrátkami na obou koncích táhne průměr dolů.

## Doporučené další kroky

1. NO-POSE přes periodicitu pohybu místo pěny (viz verdikt C).
2. Generalizace: stejný protokol na test4 / jiném bazénu.
3. Pose estimation (RTMPose) na perspektivně narovnaném cropu, jen mimo
   NO-POSE zóny; tempo z FFT jako křížová kontrola pózy.
4. Ladit zpoždění boxu jen pokud bude vadit pose modelu (alpha 0,35→0,45).

---

# Fáze F — rozšíření na znak a prsa (do 1:46)

Běhy: cut 4 (48,2–65,7 s; znak→obrátka→prsa), cut 5 (65,7–72,5 s;
**close-up na jednoho plavce — diagnostická sonda**, očekávané selhání
geometrie lan; cíl je zdokumentovat přesný způsob selhání), cuty 6–8
(72,5–105,8 s; prsa — tempo, intra-cyklická rychlost, splývavé fáze).
Vše dráha 6, nearest 8.

Otevřené rozhodnutí: NO-POSE přes periodicitu pohybu (energie oscilací
předního okraje v pásmu záběrů, klouzavé okno ~2,5 s, práh škálovaný px/m).
Implementovat až po baseline z prsou — splývavé fáze určí práh; vedlejší
produkt = délka splývání na cyklus jako trenérská metrika.

---

# Výsledky fáze F (znak, close-up, prsa)

| Střih | Obsah | TRACKING | Poznámka |
|---|---|---|---|
| 4 (48–66 s) | znak→obrátka 100 m→prsa | **98 %** | obrátka nedetekována — viz pan-bias níže |
| 5 (66–72 s) | close-up 1 plavec | 9 % | řízené selhání, zdokumentováno |
| 6 (72–79 s) | prsa | 88 % | ok |
| 7 (79–85 s) | prsa, krátký | 60 %, 39 % LOST | SAM ztráta bez re-seedu v 5,5 s |
| 8 (85–106 s) | prsa, dlouhý | **98 %** | obrátka 150 m @ 100,7 s ✓; no_pose 75 % = splývavé fáze |

Tempo per styl (fyziologicky konzistentní ⇒ validace řetězce):
motýlek ~46–51, znak ~60–63, **prsa 39,9 c/min**.

## Klíčové zjištění: pan-bias

Poloha se měří v souřadnicích OBRAZU. Když kamera panuje s plavcem
(obrátka 100 m ve střihu 4), otočení v bazénu se v obraze neprojeví —
trajektorie klesá monotónně celým záběrem. Důsledky:
- rychlosti na panujících záběrech nesou neznámý bias rychlosti panu,
- detekce obrátky funguje jen, když se obrazový pohyb viditelně otočí
  (50 m: 29,1 s vs. oficiální 28,7 ✓; 150 m: 100,7 s ✓; 100 m: minuto ✗).

**Další hranice projektu: registrace do souřadnic bazénu** (kompenzace
panu) — ukotvit x na statické prvky: čísla drah/plata, vlajky 15 m,
texturu korálků lan (optický tok mimo vodu). Teprve pak budou rychlosti
absolutní a obrátky spolehlivé na všech záběrech.

## Close-up (střih 5) — diagnóza pro single-lane režim

Fitter vnutil 8drahový žebřík do záběru se 2–3 viditelnými lany; laneQ
0,71–0,76 to NEODHALILO (měří konzistenci fitu, ne smysluplnost).
Návrh: detekce close-upu = počet skutečně podložených lan nebo rozteč
vs. výška snímku → přepnout na tracking dominantního plavce s kalibrací
px/m z viditelné rozteče dvou lan.

## Celozávodní timeline dráhy 6 (v1)

`output/test2/race_timeline_lane6.png` — 20,5–105,8 s, 2314 snímků, mimo
close-up. Guardy proti artefaktům sešívání: dělení bloků na časech střihů
(`--split-at`), lidské tempo po obrátce (medián ≤2,6 m/s zabíjí švenky),
maskování vzorků u časových děr.

Výsledek: tempo kontinuálně přes celý závod (motýlek≈48 → znak≈62 →
prsa≈40 c/min), rychlost průměr 1,12 m/s s poctivo přiznanými dírami.
Obrátky: 150 m ✓ (100,7 s); 50 m a 100 m maskovány panem kamery —
registrace do souřadnic bazénu je jednoznačně další odemykací krok.

---

# Implementace z analýzy (1+2+3) — výsledky testů

1. **Pose zóny z certifikátů periodicity (místo pěna+kontrast):** prošlá
   tempo okna = důkaz rytmického plavání; mínus obrátky ±1,5 s, podvodní
   úseky a ne-TRACKING stavy. Test: znak 68 % → **78 % použitelných
   snímků** (vyloučené jsou nyní obrátky, ne hladina); prsa poseable 83 %
   a okolí obrátky správně 0 %. Per-frame autokorelace zavržena měřením
   (SNR: rytmus 0,21 vs. obrátka 0,09 — neseparuje).
2. **Zahuštění SAM 3 skenů u krátkých záběrů: NEGATIVNÍ výsledek.**
   12 skenů navíc na střihu 7 nenašlo nic — splývající prsař je pro
   text-detektor neviditelný; LOST zůstal ~40 % (variance mezi běhy).
   Plošné zahuštění revertováno; ponecháno levné zahuštění po prvním
   zásahu. Správné řešení do roadmapy: **cross-shot handoff** — seedovat
   záběr z poslední známé pozice předchozího záběru, bez čekání na SAM 3.
3. **`analysis/race_report.py`:** celý protokol jedním příkazem — auto
   objevení CSV, střihy z cuts CSV, degenerované bloky vyřazeny podle
   TRACK % (close-up 9 % vypadl sám), tabulka per blok + timeline PNG +
   pose zóny.

## Cross-shot handoff — implementováno, otestováno

Prior z disku (CSV téže dráhy TRACKING ≤2,5 s před střihem) povoluje seed
z JEDNOHO kvalitního zásahu (conf ≥0,75, nebo ≥0,60+YOLO). Test střih 7:
seed 82,67 s při prvním zásahu (conf 0,93) — o 0,5–1,3 s dřív než baseline.

**Definitivní závěr po 3 experimentech (baseline, zahuštění, handoff):**
LOST ~39 % na střihu 7 je limit VIDITELNOSTI splývajícího prsaře pro
zpětný SAM 2, ne seedovací taktiky. Vedlejší přínos handoffu: blok
79,2–84,6 s s TRACKING 54 % nově prochází prahem protokolu (40 %).
Pro metriky lze splývavou mezeru poctivě interpolovat (splývání je
fyzikálně koast s konstantním zpomalením) — označeno jako budoucí volba.

## Single-lane režim pro close-upy — implementováno, otestováno

Detektor (dvoupodmínkový, kalibrovaný měřením): žebřík selhává (geometrie
None nebo ≤6 přímých podpor ve ≥3/4 sond) **a zároveň** málo kandidátních
lan (medián ≤12; skutečný close-up 10,5 vs. těžký wide 14). První verze
s jedinou podmínkou falešně chytila střih 7 — odhaleno regresí na všech
střizích, opraveno; finální detektor 7/7.

Chování CU: dominantní SAM 3 seed (největší box, conf ≥0,5, plocha ≥1,5 %
snímku, první zásah), tracking bez žebříku/lane guardů/verifikací, výstup
značen laneCU (protokol dráhy ho nikdy nesloučí — identita dráhy je
v detailu neověřitelná).

Výsledky: střih 5 TRACKING **9 % → 95 %**, LOST 27 % → 0 %. Regrese
širokého záběru (střih 3): 80/20/0 — beze změny proti baseline.
