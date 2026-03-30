# Forschungshistorie Paper 2 v2

## Titel
**Paper 2 – Messung als Randkopplung: Robin-basierte Doppelspalt-Studie**

## Funktion dieses Strangs
Dieser Strang baut **auf Paper 1 / Repo 1** auf, beginnt aber **nicht** sofort mit dem Doppelspalt. Zuerst wird die **Kernfunktion des Helmholtz–Robin-Feldes** weiter freigelegt, gehärtet und reduziert. Erst danach wird der Doppelspalt **schrittweise** als Simulations- und Belastungstest aufgebaut.

---

## Phase A – Problemverschiebung nach Paper 1
Nach dem ersten Repo-/Paper-Block wird die Fragestellung verschoben: Nicht sofort der Doppelspalt soll „simuliert werden“, sondern zuerst die **feldinterne Ordnungsfunktion** so weit geklärt werden, dass ein Doppelspaltaufbau darauf sauber aufsetzen kann.

Kernidee dieser Phase:
- Anschluss an den in Paper 1 freigelegten Helmholtz–Robin-Kern
- neue Arbeitsformulierung für Paper 2
- Fokus auf **Ordnungsfamilien**, nicht auf fragile Feinetiketten
- Doppelspalt zunächst **zurückstellen**, bis der Feldträger selbst sauber verstanden ist

---

## Phase B – β-Phasen und Familienhärtung
Im ersten großen Testblock wird die Feldordnung entlang eines dichten β-Scans verfolgt. Branch-tracking, Konsensauswertung und mehrere Robustheitstests zeigen:
- eine **robuste Einachsen-Grundphase** bei kleinem β
- ein **enges Reorganisationsfenster** um β ≈ 1.5–2
- danach eine **Mehrachsen-/Mehrfamilienphase**

Wichtig ist die historische Präzisierung:
Die belastbare Ordnung sitzt **auf Familienebene**, nicht auf exakten X/Y/Z-Einzellabels.

Dazu gehören:
- dichter β-Scan mit branch tracking
- Degenerations-Unterraum-Robustheit
- Readout-Robustheit
- integrierte Robustheitsbilanz

Diese Phase schließt die erste Lücke zwischen Paper 1 und Paper 2: Robin wirkt nicht nur amplituden- oder spektralverändernd, sondern als **Selektor von Ordnungsfamilien**.

---

## Phase C – Nullmodelle und phasensensitive Schärfung
Danach folgt eine methodische Verschärfung:
- Phase gegen Nullmodell / „Gegenwelt“
- phasensensitive Zusatzdiagnose
- Übersetzung bildhafter Hypothesen in prüfbare Größen

Die historische Pointe dieser Phase:
- Zell-Shuffle bzw. harte Gegenwelten zerstören die reale Phasenlesung weitgehend
- die phasensensitive Zusatzdiagnose zeigt ein echtes Signal, besonders in der Niedrig-β-Phase
- zugleich wird offen sichtbar, wo die Befunde noch nicht „Knockout“-scharf sind

Die Historie bleibt hier bewusst ehrlich: Nicht jede frühere Intuition trägt vollständig, aber die **phasenrelevante Ordnungsstruktur** trägt genug, um weiter präzisiert zu werden.

---

## Phase D – ParaView-Intuitionen werden zu Testreihen
Frühere Visualisierungen werden nicht als Evidenz behandelt, sondern als **Hypothesengenerator**.

Daraus entstehen neue Testreihen zu:
- Opposite-Locking
- Carrier-Schalen
- axialen Brücken
- Cross-layer-Alignment
- Corner–Edge–Face-Kopplung (CEF)

Methodischer Grundsatz:
- keine alten Schwellen, Radien oder Hüllenlagen übernehmen
- keine Builder-Daten rückimportieren
- nur prüfen, ob **dieselben geometrischen Auffälligkeiten** im Helmholtz–Robin-Feld **emergent** wieder auftreten

Diese Übersetzungsphase ist zentral, weil sie die alten Bilder in **modellinterne, numerisch prüfbare Aussagen** überführt.

---

## Phase E – Carrier-, Opposite- und CEF-Tests
Die eigentlichen Testblöcke zu den Visualisierungshypothesen liefern mehrere Verdichtungen:
- gegenüberliegende Zonen koppeln sich direkt (**Opposite-Locking**)
- Carrier sitzen ab bestimmten β-Bereichen klarer auf Trägerschalen
- Corner, Edge und Face erscheinen nicht als dekorative Shells, sondern als **inzidenzgebundenes Netzwerk**
- die Ordnung wird besser als **verriegeltes Netzwerk** denn als bloße lineare Kette lesbar

Wichtig ist die Korrektur der Lesart:
Der starke Befund ist **nicht** zwingend „Zentrumsmittlung“, sondern die **direkte Opposite-Kopplung** und ein stabiler CEF-Backbone.

---

## Phase F – Reduktionsstrategie zum Kernprinzip
Nach der Vielzahl der Diagnosen beginnt eine bewusste Reduktionsstrategie.

Ziel:
- aus vielen geometrischen Indikatoren einen **kleinen tragenden Kern** herausziehen
- dabei zwischen Backbone-Struktur und phasenabhängigen Modulatoren unterscheiden

Historische Verdichtung:
- ein kleiner Merkmalskern erreicht bereits starke Trennleistung
- `mean_boundary`, `center_frac`, `opp_edge` erscheinen als sehr kompakter 3er-Kern
- dazu tritt ein stabiler Backbone aus `CF_inc`, `CE_inc`, `EF_inc`, `L_align`
- `C_opp`, `E_opp`, `F_opp` verhalten sich eher als phasenabhängige Modulatoren

Zugleich bleibt offen:
Der Carrier-Graph scheint **notwendig**, ist aber für die genaue Brückenform noch nicht vollständig hinreichend.

---

## Phase G – Kernfunktion des Helmholtz–Robin-Feldes
Der letzte Vorblock vor dem Doppelspalt zieht die Befunde auf die eigentliche Feldkernfunktion zusammen.

Die vorläufige Kernlesung lautet:
- **geschlossener gebundener Kern**
- **carriertragender Übergangs-/Schalenbereich**
- **shell/boundary-bias**
- **Opposite-Locking**
- **stabiler CEF-Backbone**
- Brücken eher als **sekundäre Kopplungsfolge** als als primärer Selektor

Methodische Schlussregel:
Die einfachste Kernstruktur des Helmholtz–Robin-Feldes soll **nur aus dem Feld selbst** freigelegt werden. Alte Builder-/ParaView-Bilder dienen höchstens als geometrische Hinweisgeber.

---

## Phase H – Beginn des eigentlichen Doppelspaltaufbaus
Erst nach dieser Vorarbeit startet der Doppelspaltblock.

Der Aufbau wird bewusst **stufenweise** organisiert:
1. **Einspalt-Baseline**
2. **Doppelspalt ohne Messstörung**
3. erste Messstörungs-Surrogate
4. Ausschluss zu einfacher Robin-Deutung
5. realistisches Detektormodell
6. präregistrierte Detektormatrix
7. Minimalformel
8. externe Validierung und Holdout
9. halb-analytische Herleitung
10. mikroskopisches Robin-Rauschmodell

---

## Phase I – Baselines des Doppelspalts
Die Simulationskette beginnt mit den einfachsten Referenzfällen:
- **ein Spalt** → einzelne zentrale Beugungskeule
- **zwei Spalte ohne Messstörung** → symmetrisches Fransenmuster mit hoher Sichtbarkeit

Das fixiert die Referenz, bevor Detektoreffekte eingeführt werden.

---

## Phase J – Messsurrogate und Ausschluss einfacher Robin-Lesungen
Danach werden mehrere Mechanismen gegeneinander getestet:
- konstantes Robin
- statisches räumliches Robin-Profil
- stochastische Robin-Störung
- Which-way / Path-marking
- kleiner Verlustkanal

Historische Verdichtung:
- **konstantes/statisches Robin allein** verzerrt und asymmetriert das Muster, löscht die Fransen aber nicht sauber
- **stochastische Robin-Störung** senkt die Sichtbarkeit deutlich
- **Path-marking** senkt die Sichtbarkeit ebenfalls, bei nahezu erhaltener Transmission
- ein kleiner Verlustkanal verstärkt den Effekt, ist aber nicht der primäre Kern

Die zentrale Trennung ist hier erreicht:
**Musterverzerrung** ist nicht dasselbe wie **Kohärenzverlust**.

---

## Phase K – Realistisches Detektormodell
Aus den Ausschlusstests wird ein realistischerer Detektorkern formuliert:
- lokale Robin-Randänderung
- unaufgelöste Fluktuation / Störkopplung
- möglicher Which-way-Marker
- ggf. kleiner Verlustkanal

Der entscheidende Befund dieser Phase:
Ein realer Detektor lässt sich im Modell **nicht** auf „Spalt wird zugedrückt“ reduzieren. Stattdessen bleibt die Apertur offen, während die **Interferenzkomponente** gezielt unterdrückt wird.

---

## Phase L – Präregistrierte Detektormatrix und Minimalformel
Die stärkste Verdichtung des Doppelspaltblocks ist die minimale Detektorformel

\[
V(\mu,\sigma)=V_0\,\mu\,e^{-a\sigma-b\sigma^2}.
\]

Lesung:
- \(\mu\): Path-marking-Überlappung des Detektorzustands
- \(\sigma\): effektive Stärke der Robin-Fluktuation
- \(V_0\): kohärente Basis-Sichtbarkeit

Inhaltlich bedeutet das:
- Path-marking senkt die Kohärenz **linear** über \(\mu\)
- Robin-Fluktuation senkt sie **nichtlinear** über \(\exp(-a\sigma-b\sigma^2)\)
- beide greifen am **Interferenzterm**, nicht primär an der Gesamtdurchlässigkeit

Wichtig für die Historie:
Diese Formel wird nicht einfach postuliert, sondern als **kleinster effektiver Detektorkern** nach mehreren stärkeren Kandidaten herausgeschält.

---

## Phase M – Externe Validierung und Holdout
Die Minimalformel wird anschließend gehärtet:
- externe Path-marking-Tests
- externe Sigma-only-Tests
- kombinierte externe Validierung
- strenger Holdout auf ungesehenen Kombinationen

Damit ist die Formel nicht nur ein Fit auf der Matrix, sondern eine **generaliserende Gesetzesform innerhalb des Modells**.

---

## Phase N – Halb-analytische und mikroskopische Verdichtung
Danach folgt die theoretische Verdichtung in zwei Schritten:

### 1. Halb-analytische Begründung
Der Detektor greift am Kreuzterm der Zwei-Spalt-Intensität an:
- Detektorzustandsüberlappung liefert den linearen \(\mu\)-Faktor
- Robin-Mikrostörung wirkt als effektive Phasenverwaschung
- daraus folgt die exponentielle Unterdrückung

### 2. Mikroskopisches Robin-Rauschmodell
Getestet werden:
- reines Phasenrauschen
- reine Betrags-/Impedanzstörung
- gemischtes Robin-Mikromodell

Befund:
Nur die **Mischung** reproduziert gleichzeitig den linearen und quadratischen Term der empirischen Minimalformel in sehr guter Näherung.

Damit erhält die Minimalformel erstmals einen **konkreten mikroskopischen Kandidaten**.

---

## Zwischenfazit der gesamten Paper-2-Historie bis hierher
Paper 2 beginnt **nicht** mit dem Doppelspalt, sondern mit der Freilegung eines feldinternen Ordnungs- und Trägerkerns. Erst aus dieser Vorarbeit heraus wird der Doppelspalt schrittweise aufgebaut.

Die bisherige Gesamtverdichtung lautet:
- das Helmholtz–Robin-Feld besitzt eine eigene Kernfunktion mit gebundenem Kern, carriertragender Schale, Opposite-Locking und CEF-Backbone
- der Doppelspalt ist der erste große Belastungstest dieses geklärten Feldrahmens
- eine reine statische Robin-Deutung genügt nicht
- ein realistischer Detektor wirkt im Modell minimal als **Wegmarkierung + unaufgelöste Robin-Störung**
- die resultierende Minimalformel generalisiert und erhält sogar einen mikroskopischen Robin-Rauschkandidaten

Damit schließt Paper 2 die Lücke zwischen dem geometrischen Kern aus Paper 1 und einer messungsnahen Doppelspalt-Simulation.
