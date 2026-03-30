# Repo-Unterstruktur Paper 2 v1

## Ziel des Repos
Repo 2 baut auf **Paper 1 / Repo 1** auf, wiederholt dessen Kern aber nicht vollständig. Es dokumentiert den Weg von der **feldinternen Kernfunktion** bis zur **Doppelspalt-/Detektor-Minimalformel**.

---

## Empfohlene Top-Level-Struktur

```text
paper2-repo/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── docs/
├── scripts/
├── src/
├── results/
├── figures/
└── archive/
```

---

## Empfohlene inhaltliche Struktur

### 00_foundation_from_paper1/
**Funktion:** Anschluss an Repo 1.

Inhalt:
- kurze Anschlussnotiz: was aus Paper 1 übernommen wird
- welche Begriffe und Diagnosen hier vorausgesetzt werden
- was **nicht** noch einmal bewiesen wird

### 01_field_core_ordering/
**Funktion:** Problemverschiebung und frühe Härtung des Feldkerns.

Inhalt:
- neue Fragestellung / interne Arbeitsformulierung
- Anschluss an Paper 1
- dichter β-Scan mit branch tracking
- Degenerations-Unterraum-Robustheit
- Readout-Robustheit
- integrierte Robustheitsbilanz

### 02_nullmodels_and_phase_sensitivity/
**Funktion:** Gegenwelt- und Phasenschärfung.

Inhalt:
- Nullmodellblock / Gegenwelt
- phasensensitive Zusatzdiagnose
- Übersetzung beobachteter Hypothesen in quantitative Tests

### 03_carrier_opposite_cef_tests/
**Funktion:** Übersetzung der alten Visualisierungshypothesen in echte Geometrietests.

Inhalt:
- Opposite-locking
- Carrier-Schalen
- axiale Brücken
- Cross-layer-alignment
- CEF-Inzidenztests

### 04_reduction_to_minimal_core/
**Funktion:** Reduktionsstrategie A/B/C.

Inhalt:
- kompakte Kernfeatures
- Backbone vs. Modulatoren
- Minimal-Kern
- Carrier-Graph
- offene Restlücken

### 05_field_core_function/
**Funktion:** vorläufig endgültige Formulierung der Kernfunktion des Helmholtz–Robin-Feldes.

Inhalt:
- geschlossener gebundener Kern
- carriertragender Übergangsbereich
- shell/boundary-bias
- Opposite-locking
- CEF-backbone
- methodische Regel: nur feldinterne emergente Struktur lesen

### 06_double_slit_baselines/
**Funktion:** Beginn der eigentlichen Doppelspaltkette.

Inhalt:
- Einspalt-Baseline
- Doppelspalt ohne Messstörung
- Feldbilder und Schirmprofile
- Referenzmetriken (Visibility, Transmission, Fransenabstand)

### 07_measurement_surrogates_and_robin_tests/
**Funktion:** erste Messsurrogate und Ausschluss einfacher Robin-Lesungen.

Inhalt:
- dephasing-/Robin-ähnliche Surrogate
- konstantes Robin
- statisches räumliches Robin-Profil
- stochastische Robin-Störung
- Path-marking
- Verlustkanal
- Vergleich: Verzerrung vs. Kohärenzverlust

### 08_real_detector_model/
**Funktion:** realistischere Detektorbeschreibung.

Inhalt:
- \(\Delta\beta\) + Inhomogenität + Fluktuation + kleiner Verlustkanal
- Path-marking-Kopplung
- Vergleich zur inkohärenten Referenz
- getrennte Auswertung von Visibility und Transmission

### 09_detector_matrix_and_minimal_formula/
**Funktion:** präregistrierte Matrix und empirische Verdichtung.

Inhalt:
- Detektormatrix
- Fit der Minimalformel
- externe Validierung
- Holdout-Test
- zentrale Ergebnisplots und Tabellen

### 10_microscopic_robin_noise_model/
**Funktion:** theoretische Verdichtung.

Inhalt:
- halb-analytische Herleitung aus dem Kreuzterm
- Phasen-/Impedanzbild der Robin-Störung
- Mikromodell: Phase-only, amplitude-only, mixed
- Reproduktion der Koeffizienten \(a,b\)
- segmentierte / korrelierte Spaltmodelle

### 11_paper_maps/
**Funktion:** Abstimmung von Repo und Manuskript.

Inhalt:
- claim map
- section map
- figure map
- methods map

### archive/
**Funktion:** Provenienz und Forschungshistorie.

Inhalt:
- geordnete Chatexports
- Forschungshistorie Paper 2
- historische Skripte / Rohnotizen
- Kennzeichnung rekonstruiert vs. original

---

## Empfohlene Dokumente unter docs/

- `docs/forschungshistorie_paper2.md`
- `docs/repo_map.md`
- `docs/reproducibility_status.md`
- `docs/provenance_and_reconstructions.md`
- `docs/paper2_claim_map.md`
- `docs/paper2_section_map_arxiv.md`
- `docs/paper2_figure_map.md`

---

## Empfohlene public-release-Regel

Im öffentlichen Repo soll **klar** erkennbar sein:
- Paper 2 beginnt **methodisch** mit dem Feldkern, nicht mit dem Doppelspalt
- der Doppelspaltblock ist **schrittweise** aufgebaut
- die Minimalformel ist **effektiv / minimal**, nicht als letzte Mikrophysik ausgegeben
- rekonstruierten Dateien steht ihr Status offen bei

---

## Empfohlene ArXiv-Schnittstelle

### In den Main Text von Paper 2
- kurzer Verweis auf Paper 1 / Repo 1
- Kernfunktion des Feldes nur knapp als Motivation
- Einspalt / Doppelspalt-Baselines
- Ausschluss einfacher Robin-Erklärungen
- realistisches Detektormodell
- Minimalformel
- Generalisierung
- mikroskopische Robin-Rauschverdichtung

### In Appendix / Supplement
- ausführliche β-Phasen- und Carrier-Härtung
- zusätzliche Opposite-/CEF-Tests
- erweiterte Matrixplots
- technische Reproduktionsdetails

### Nur im Repo
- komplette Forschungshistorie
- Roh- und Zwischenläufe
- alle Zusatzdiagnosen
- volle Provenienzschicht

---

## Kernsatz des Repos
Dieses Repo dokumentiert die Brücke

**von der Kernfunktion des Helmholtz–Robin-Feldes**

zu

**einer schrittweise gehärteten Doppelspalt-/Detektor-Minimalformel**.
