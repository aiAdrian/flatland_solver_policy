# Flatland Multi-Agent Observations – Übersicht & Feature-Design

Diese Dokumentation beschreibt alle wichtigen Beobachtungsklassen (Observations) für Multi-Agenten-Umgebungen im Flatland-Railway-Setting. Sie legt besonderen Fokus auf die Feature-Struktur, Entscheidungslogik und die Unterschiede zwischen den Klassen.

---

## 1. ExperimentalObservation

**Beschreibung:**
Die `ExperimentalObservation` ist eine generische, 30-dimensionale Beobachtung für jeden Agenten. Sie kombiniert Agenten-Status (Position, Richtung, Ziel, Status) mit einer Analyse der Umgebung und anderer Agenten. Sie ist die Basisklasse für komplexere, temporale oder multi-agentenfähige Beobachtungen.

**Feature-Design:**
- 30-dimensionale Feature-Vektoren (Details siehe Code)
- Enthält keine explizite Entscheidungslogik für Weichen/Merges, sondern gibt eine allgemeine Zustandsbeschreibung zurück.

**Einsatz:**
- Basis für temporale und multi-agentenfähige Observations
- Gut geeignet für klassische RL-Algorithmen

---


## 2. DecisionPointObservation

Die `DecisionPointObservation` ist ein spezialisierter Beobachtungs-Builder für Multi-Agenten-Umgebungen im Flatland-Railway-Setting. Sie liefert für jeden Agenten einen 42-dimensionalen Feature-Vektor, der die wichtigsten Entscheidungssituationen im Bahnnetz abbildet: **Start**, **Weiche (Switch)**, **Merge/Crossing**. Die Features sind disjunkt angeordnet und werden nur für die jeweils relevante Situation befüllt, alle anderen Felder bleiben 0. Die Feature-Logik und -Befüllung ist eng an die Flatland-rl-API und die reale Entscheidungsstruktur im Schienennetz angelehnt.

### Kontext & Zielsetzung

- **Flatland-rl** ist eine Multi-Agenten-Umgebung zur Simulation von Zugverkehr und Konfliktlösung auf Schienennetzen. Die DecisionPointObservation abstrahiert die wichtigsten Entscheidungspunkte (Start, Weiche, Merge/Crossing) in einen RL-tauglichen, strukturierten Feature-Vektor.
- Die Features werden dynamisch und kontextsensitiv befüllt: Nur die für die aktuelle Entscheidungssituation relevanten Blöcke sind aktiv, alle anderen sind 0.
- Die Entscheidungslogik basiert auf rekursiver Tiefensuche (DFS) zur Pfadanalyse, Deadlockerkennung und Zielprüfung.


### Feature-Tabelle: DecisionPointObservation (alle 42 Features einzeln)

| Index | Name                        | Beschreibung                                                                 | decision_type (gesetzt bei)         |
|-------|-----------------------------|------------------------------------------------------------------------------|-------------------------------------|
| 0     | decision_type               | Entscheidungssituation (0=normal, 1=Start, 2=Switch, 4=Merge, 8=DONE)        | alle                                |
| 1     | onehot_left                 | 1, wenn links optimal (laut distance_map), sonst 0                           | Switch, Merge/Crossing              |
| 2     | onehot_forward              | 1, wenn geradeaus optimal, sonst 0                                           | Switch, Merge/Crossing              |
| 3     | onehot_right                | 1, wenn rechts optimal, sonst 0                                              | Switch, Merge/Crossing              |
| 4     | left_curr_dist              | Aktuelle Distanz (vor Schritt links)                                         | Switch                              |
| 5     | left_deadlock               | Deadlock-Flag nach Schritt links                                             | Switch                              |
| 6     | left_switches               | Anzahl Weichen nach Schritt links                                            | Switch                              |
| 7     | left_delta_dist             | Distanzdifferenz zum Ziel nach Schritt links                                 | Switch                              |
| 8     | left_target_found           | Ziel erreicht nach Schritt links                                             | Switch                              |
| 9     | left_abort                  | Abbruch-Flag nach Schritt links                                              | Switch                              |
| 10    | forward_curr_dist           | Aktuelle Distanz (vor Schritt geradeaus)                                     | Switch                              |
| 11    | forward_deadlock            | Deadlock-Flag nach Schritt geradeaus                                         | Switch                              |
| 12    | forward_switches            | Anzahl Weichen nach Schritt geradeaus                                        | Switch                              |
| 13    | forward_delta_dist          | Distanzdifferenz zum Ziel nach Schritt geradeaus                             | Switch                              |
| 14    | forward_target_found        | Ziel erreicht nach Schritt geradeaus                                         | Switch                              |
| 15    | forward_abort               | Abbruch-Flag nach Schritt geradeaus                                          | Switch                              |
| 16    | right_curr_dist             | Aktuelle Distanz (vor Schritt rechts)                                        | Switch                              |
| 17    | right_deadlock              | Deadlock-Flag nach Schritt rechts                                            | Switch                              |
| 18    | right_switches              | Anzahl Weichen nach Schritt rechts                                           | Switch                              |
| 19    | right_delta_dist            | Distanzdifferenz zum Ziel nach Schritt rechts                                | Switch                              |
| 20    | right_target_found          | Ziel erreicht nach Schritt rechts                                            | Switch                              |
| 21    | right_abort                 | Abbruch-Flag nach Schritt rechts                                             | Switch                              |
| 22    | merge_deadlock_fwd          | Deadlock-Flag nach Schritt vorwärts (Merge/Crossing)                         | Merge/Crossing (forward)            |
| 23    | merge_switches_fwd          | Anzahl Weichen nach Schritt vorwärts (Merge/Crossing)                        | Merge/Crossing (forward)            |
| 24    | merge_target_found_fwd      | Ziel erreicht nach Schritt vorwärts (Merge/Crossing)                         | Merge/Crossing (forward)            |
| 25    | merge_abort_fwd             | Abbruch-Flag nach Schritt vorwärts (Merge/Crossing)                          | Merge/Crossing (forward)            |
| 26    | merge_deadlock_bwd          | Deadlock-Flag nach Schritt rückwärts (Merge/Crossing)                        | Merge/Crossing (backward)           |
| 27    | merge_switches_bwd          | Anzahl Weichen nach Schritt rückwärts (Merge/Crossing)                       | Merge/Crossing (backward)           |
| 28    | merge_target_found_bwd      | Ziel erreicht nach Schritt rückwärts (Merge/Crossing)                        | Merge/Crossing (backward)           |
| 29    | merge_abort_bwd             | Abbruch-Flag nach Schritt rückwärts (Merge/Crossing)                         | Merge/Crossing (backward)           |
| 30    | agent_state_ready_to_depart | One-hot: Agent ist READY_TO_DEPART                                           | alle                                |
| 31    | agent_state_malfunction_off_map | One-hot: Agent ist MALFUNCTION_OFF_MAP                                  | alle                                |
| 32    | agent_state_moving          | One-hot: Agent ist MOVING                                                    | alle                                |
| 33    | agent_state_stopped         | One-hot: Agent ist STOPPED                                                   | alle                                |
| 34    | agent_state_malfunction     | One-hot: Agent ist in MALFUNCTION                                            | alle                                |
| 35    | agent_state_done            | One-hot: Agent ist DONE                                                      | alle                                |
| 36    | agent_state_other           | One-hot: Sonstiger Status                                                    | alle                                |
| 37    | last_action_left            | One-hot: Letzte Aktion war links                                              | alle                                |
| 38    | last_action_forward         | One-hot: Letzte Aktion war geradeaus                                         | alle                                |
| 39    | last_action_right           | One-hot: Letzte Aktion war rechts                                            | alle                                |
| 40    | last_action_stop            | One-hot: Letzte Aktion war stop                                              | alle                                |
| 41    | last_action_other           | One-hot: Letzte Aktion war sonstiges                                         | alle                                |

### Feature-Befüllung & Speziallogik

- **decision_type** wird immer gesetzt und kodiert die aktuelle Entscheidungssituation.
- Die Blöcke für Switch und Merge/Crossing werden nur bei passender Situation befüllt, alle anderen Felder bleiben 0.
- Die Features für jede Richtung werden per rekursiver DFS mit Backtracking berechnet (`_navigate_direction`), inkl. Deadlock- und Cycle-Erkennung.
- Unerreichbare Zellen (`np.inf`) werden durch $2 \times$ aktuelle Distanz ersetzt, NaN/Inf werden durch -1 ersetzt.
- Deadlocks werden erkannt, wenn ein entgegenkommender Agent auf dem Pfad ist.
- **abort**-Flags werden gesetzt, wenn die maximale Schrittzahl (max_steps) überschritten wird.
- **target_found** wird gesetzt, wenn das Ziel auf dem Pfad erreicht wird.


### Entscheidungsfindung & DFS-Logik (Algorithmus-Details)

Die Methode `_navigate_direction` ist das Herzstück der Entscheidungslogik. Sie implementiert eine rekursive Tiefensuche (Depth-First Search, DFS), um für jede relevante Richtung (links, geradeaus, rechts, rückwärts) den Pfad zum Ziel zu analysieren und dabei Deadlocks, Weichen, Zielerreichung und Abbruchbedingungen zu erkennen.


#### Algorithmus in verständlicher Prosa

Die rekursive Tiefensuche (DFS) zur Entscheidungsfindung funktioniert wie folgt:

1. **Start und Zielprüfung:** Die Suche beginnt an der aktuellen Position und prüft zunächst, ob das Ziel bereits erreicht ist. Ist dies der Fall, wird dies sofort als Erfolg gemeldet.
2. **Abbruchbedingungen:** Die Suche wird abgebrochen, wenn eine maximale Suchtiefe überschritten wird (um endlose Rekursionen zu verhindern) oder wenn die aktuelle Position und Richtung bereits besucht wurden (Zyklenerkennung).
3. **Deadlock-Erkennung:** Trifft die Suche auf einen entgegenkommenden oder blockierenden Agenten, wird dies als Deadlock erkannt und entsprechend im Ergebnis markiert.
4. **Weichen und Alternativen:** An jeder Weiche (Switch) prüft die Suche alle möglichen Weiterführungen (z.B. links, geradeaus, rechts, rückwärts). Für jede Alternative wird die Suche rekursiv fortgesetzt. So werden alle potenziellen Pfade analysiert.
5. **Backtracking und Auswahl:** Nach der Analyse aller Alternativen wählt die Suche das beste Ergebnis aus (z.B. den Pfad mit minimalem Deadlock-Risiko oder kürzester Distanz).
6. **Feature-Befüllung:** Die Ergebnisse der Suche – Distanz zum Ziel, Deadlock-Status, Anzahl durchlaufener Weichen, Zielerreichung und Abbruch-Flag – werden für jede Richtung in die entsprechenden Features des Beobachtungsvektors eingetragen.
7. **Cycle Prevention:** Um Endlosschleifen zu vermeiden, merkt sich die Suche alle bereits besuchten (Position, Richtung)-Paare und prüft vor jedem Schritt, ob sie erneut betreten werden.

Diese Vorgehensweise stellt sicher, dass alle relevanten Entscheidungsalternativen im Schienennetz berücksichtigt werden, Deadlocks und Konflikte realistisch erkannt werden und die resultierenden Features robust und RL-tauglich sind.

#### Warum ist diese DFS-Logik ideal für Flatland-RL?

- **Realistische Konflikterkennung:** Die DFS erkennt echte Deadlocks und Konflikte mit anderen Agenten, was für Multi-Agenten-Planung und RL-Policies essenziell ist.
- **Flexible Pfadanalyse:** Durch Backtracking an Weichen werden alle Alternativen geprüft, sodass Policies nicht in lokale Minima laufen und komplexe Entscheidungssituationen abgebildet werden.
- **Effizient und sicher:** Die maximale Suchtiefe verhindert endlose Rekursionen und hält die Berechnung effizient und RL-tauglich.
- **Zyklenerkennung:** Cycle Prevention ist wichtig, da das Schienennetz Zyklen enthalten kann – so werden Endlosschleifen vermieden.
- **Konfliktlösung:** Die DFS speichert alle gesehene Agenten auf dem Pfad, was für Prioritätsentscheidungen und Konfliktlösung genutzt werden kann.
- **RL-taugliche Features:** Die so gewonnenen Features (Distanz, Deadlock, Ziel, Weichen, Abbruch) sind direkt für RL nutzbar und ermöglichen Policies, auf komplexe Situationen zu reagieren.
- **Disjunkte Feature-Blöcke:** Nur die für die aktuelle Entscheidungssituation relevanten Features werden befüllt, was die Policy-Entwicklung vereinfacht und Overfitting reduziert.

**Fazit:**
Die rekursive DFS-Logik in `_navigate_direction` bildet die reale Entscheidungsstruktur im Flatland-Setting ab, erkennt Deadlocks und Konflikte, prüft alle Alternativen und liefert robuste, RL-taugliche Features für jede relevante Richtung. Damit ist sie optimal geeignet, um Policies für komplexe Multi-Agenten-Szenarien im Bahnnetz zu trainieren.

### Flatland-Kontext & RL-Tauglichkeit

- Die DecisionPointObservation abstrahiert die komplexen Entscheidungssituationen im Flatland-Setting in einen RL-tauglichen, disjunkten Feature-Vektor.
- Sie erkennt und kodiert alle relevanten Entscheidungs- und Konfliktpunkte, sodass Policies gezielt auf diese Situationen reagieren können.
- Die Beobachtung ist effizient, flexibel und für Einzel- wie Multi-Agenten-Szenarien geeignet.

### Methodenüberblick

- `get(handle)`: Erzeugt den Feature-Vektor für einen einzelnen Agenten.
- `get_many(handles)`: Erzeugt die Beobachtungen für mehrere Agenten gleichzeitig.
- `_shortest_path_action_hint(...)`: Berechnet, welche Richtung (links, geradeaus, rechts) entlang des kürzesten Pfads zum Ziel optimal ist.
- `_navigate_direction(...)`: Führt die rekursive Tiefensuche durch, um Pfadmetriken, Deadlocks und Zielerreichung zu bestimmen.

---

## 3. SimplifiedPathThreeTierObservation

**Beschreibung:**
Die `SimplifiedPathThreeTierObservation` teilt die Umgebung eines Agenten in drei Pfadsegmente ("Tiers"): links, geradeaus, rechts. Für jede Richtung werden identische Feature-Blöcke berechnet (z.B. Distanz, Deadlock, Zielrichtung). Ein Header enthält State-Informationen und einen one-hot-Hinweis auf den besten Pfad.

**Feature-Design:**
- Header (z.B. State, Richtung, Position, Ziel, Agentenzahl, one-hot best path)
- Drei Blöcke à N Features (z.B. 16) für [links, geradeaus, rechts]

**Einsatz:**
- Gut geeignet für Policies, die explizit zwischen Alternativen wählen
- Übersichtliche, strukturierte Feature-Vektoren

---

## 4. Temporale Multi-Agenten-Observations

### 4.1 TemporalMultiAgentObservation
**Beschreibung:**
Kombiniert die Beobachtungen mehrerer Agenten über mehrere Zeitschritte (z.B. T=3). Jeder Agent erhält eine Historie seiner eigenen Beobachtungen (und ggf. Velocity-Features). Besonders nützlich für Algorithmen mit zeitlichen Abhängigkeiten (z.B. RNNs).

**Feature-Design:**
- Pro Zeitschritt: 30D Basis-Features + 3D Velocity (velocity_x, velocity_y, angular_velocity)
- Rückgabeformat: Liste von (obs_t-2, obs_t-1, obs_t) pro Agent

### 4.2 TemporalMultiAgentSwitchObservation
**Beschreibung:**
Fokussiert auf Weichen (Switches) im Schienennetz. Enthält explizite Informationen über Weichenpositionen und -zustände über mehrere Zeitschritte.

### 4.3 TemporalMultiAgentSwitchCellObservation
**Beschreibung:**
Erweitert die vorherige um Informationen zu Zellen, die zu Weichen führen oder in deren Nähe liegen. Unterstützt Überholen und Konfliktlösung.

### 4.4 TemporalMultiAgentSwitchCellWithDirectionObservation
**Beschreibung:**
Ergänzt die SwitchCellObservation um Richtungsinformationen. Agenten wissen, in welche Richtung sie sich bewegen und welche Abzweigungen möglich sind.

---

## Zusammenfassung

Alle Observations sind darauf ausgelegt, die Entscheidungsfindung der Agenten in komplexen, dynamischen Schienennetzen zu verbessern. Die DecisionPointObservation bietet dabei die fortschrittlichste Entscheidungslogik mit disjunkten, klar nummerierten Feature-Blöcken für alle relevanten Entscheidungstypen.