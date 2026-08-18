
---

# Fáze 6 — registrace do souřadnic bazénu: výsledky výzkumu

## Krok 1 (brána): korálková fáze lan — ZAMÍTNUTO měřením
Broadcast 1080p korálky rozmazává: cross-korelace profilů mezi snímky bez
zámku (ostrost ~1,0), posuny nekonzistentní mezi lany (−22 vs. −91 px),
u obrátky 100 m detekce lan zcela vypadává.

## Pivot na optický tok mimo vodu — mechanismus VALIDOVÁN
- statický start: 0,00 px; panující úseky MAD < 2 px; funguje i u obrátky
  100 m (148–161 bodů), kde lana selhala,
- syntetický pan ±5..±40 px obnoven s chybou 0,00 px,
- camera_track pro 20,5–105,8 s: 2560 snímků, 91 % confident,
  změřený pan ±4 000 px (dvě šířky obrazu!) uvnitř záběrů,
- nejisté úseky jen v close-upu (65,8–72,5) — mimo protokol.

## Integrace odhalila dvě chyby (opraveny)
1. Poloha/lokální px/m je neplatná konverze — s offsety ±2 000 px vyrábí
   každé zakolísání ppm falešné metry; poloha se musí INTEGROVAT z
   přírůstků (v_px/ppm·dt).
2. Směr a integrace používaly nemaskované v_px (vč. PREDICTED koastů a
   špiček) — poloha šplhala 2,9 m/s tam, kde rychlost hlásila 0,07.

## Definitivní verdikt: translační korekce NESTAČÍ u stěn
Vizuální ground truth (150 m obrátka): stěna leží u úběžníku lan —
px/m PODÉL bazénu tam klesá několikanásobně pod příčné měřítko z rozteče
lan. Pan v px pak přebije skutečný pohyb a otočku SMAŽE místo odhalení.
Test „obrátka 100 m se objeví" NEPROŠEL. Korekce je proto OPT-IN
(`race_report --use-camera`), nikdy tichý default; bez ní je chování
identické stavu před fází 6 (ověřeno regresí).

## Co fáze 6b potřebuje (přesné zadání)
Plnou projektivní registraci: per-frame homografii obraz→rovina bazénu
z geometrie lan (rodina rovnoběžek se známými roztečemi 2,5 m) + jedna
kotva PODÉL bazénu (čára stěny/bloků při obrátce, 15m značky, případně
dotyk stěny z detekované obrátky). Změřený camera_track zůstává platným
vstupem (regularizace řetězení homografií). Bez homografie mají absolutní
rychlosti smysl jen ve střední části bazénu.

## Integrace pool_x do protokolu — HOTOVO

`analysis/pool_pass.py` (sidecar pool_x_lane{N}.csv; lane 6: 33 keyframů,
87 % pokrytí — díry jen close-up a cut7 bez keyframů) + `compute_speed`
s absolutní polohou (žádný px/m převod) + race_report auto-detekce.
Tempo dostává obrazový směr (pool směr přepínal vedoucí hranu — motýlek
padal na 33 c/min; opraveno, zpět 47).

První celozávodní protokol v absolutních metrech: rychlosti
0,55–1,03 m/s per blok, obrátka 150 m z reálné polohy @ 98,4 s.
Další sklizeň: tabulka mezičasů per 50 m, OCR oficiálních splitů,
maskování broadcast grafiky v camera tracku.
