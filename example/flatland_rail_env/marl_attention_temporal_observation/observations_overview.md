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

Die `DecisionPointObservation` liefert für jeden Agenten einen 34-dimensionalen Feature-Vektor, der die drei wichtigsten Entscheidungssituationen im Flatland-Setting abbildet: **Start**, **Weiche (Switch)**, **Merge/Crossing**. Die Features sind disjunkt angeordnet und werden nur für die jeweils relevante Situation befüllt, alle anderen Felder bleiben 0.

### Feature-Index und Bedeutung

| Index | Name/Bedeutung | Typ | Beschreibung | Setzlogik (decision_type) |
|-------|----------------|-----|--------------|---------------------------|
| 0     | decision_type  | float | Typ der Entscheidungssituation:<br>0 = Standard (kein Decision Point)<br>1 = Start (READY_TO_DEPART)<br>2 = Switch (Weiche, Agent kann abzweigen)<br>3 = Merge/Crossing (vor Weiche, Konfliktpotenzial)<br>-1 = DONE | Immer gesetzt |
| 1     | one-hot_left   | float | 1, wenn links der beste Pfad (laut distance_map), sonst 0 | 2, 3 |
| 2     | one-hot_forward| float | 1, wenn geradeaus der beste Pfad, sonst 0 | 2, 3 |
| 3     | one-hot_right  | float | 1, wenn rechts der beste Pfad, sonst 0 | 2, 3 |

#### Switch-Block (decision_type == 2, Weiche)
| Index | Name | Typ | Beschreibung |
|-------|------|-----|--------------|
| 4     | left_dist      | float | Distanz auf linkem Pfad zum Ziel (DFS, -1 falls nicht möglich) |
| 5     | left_deadlock  | float | 1, falls Deadlock auf linkem Pfad, sonst 0 |
| 6     | left_switches  | float | Anzahl durchlaufener Weichen auf linkem Pfad |
| 7     | left_delta_dist| float | Distanzdifferenz zum Ziel nach Schritt links (dist - curr_dist) |
| 8     | left_target_found | float | 1, wenn Ziel auf linkem Pfad erreicht, sonst 0 |
| 9     | left_abort     | float | 1, wenn max_steps auf linkem Pfad überschritten, sonst 0 |
| 10    | forward_dist      | float | Distanz auf geradem Pfad zum Ziel |
| 11    | forward_deadlock  | float | 1, falls Deadlock auf geradem Pfad |
| 12    | forward_switches  | float | Anzahl Weichen auf geradem Pfad |
| 13    | forward_delta_dist| float | Distanzdifferenz zum Ziel nach Schritt geradeaus |
| 14    | forward_target_found | float | 1, wenn Ziel auf geradem Pfad erreicht |
| 15    | forward_abort     | float | 1, wenn max_steps auf geradem Pfad überschritten |
| 16    | right_dist      | float | Distanz auf rechtem Pfad zum Ziel |
| 17    | right_deadlock  | float | 1, falls Deadlock auf rechtem Pfad |
| 18    | right_switches  | float | Anzahl Weichen auf rechtem Pfad |
| 19    | right_delta_dist| float | Distanzdifferenz zum Ziel nach Schritt rechts |
| 20    | right_target_found | float | 1, wenn Ziel auf rechtem Pfad erreicht |
| 21    | right_abort     | float | 1, wenn max_steps auf rechtem Pfad überschritten |
| 22    | reverse_dist      | float | Distanz auf rückwärts Pfad zum Ziel |
| 23    | reverse_deadlock  | float | 1, falls Deadlock auf rückwärts Pfad |
| 24    | reverse_switches  | float | Anzahl Weichen auf rückwärts Pfad |
| 25    | reverse_delta_dist| float | Distanzdifferenz zum Ziel nach Schritt rückwärts |
| 26    | reverse_target_found | float | 1, wenn Ziel auf rückwärts Pfad erreicht |
| 27    | reverse_abort     | float | 1, wenn max_steps auf rückwärts Pfad überschritten |

#### Merge/Crossing-Block (decision_type == 3)
| Index | Name | Typ | Beschreibung |
|-------|------|-----|--------------|
| 22    | fwd_deadlock_merge      | float | 1, falls Deadlock vorwärts (Merge/Crossing) |
| 23    | fwd_switches_merge      | float | Anzahl Weichen vorwärts (Merge/Crossing) |
| 24    | fwd_target_found_merge  | float | 1, wenn Ziel vorwärts erreicht (Merge/Crossing) |
| 25    | fwd_abort_merge         | float | 1, wenn max_steps vorwärts überschritten (Merge/Crossing) |
| 28    | bwd_deadlock_merge      | float | 1, falls Deadlock rückwärts (Merge/Crossing) |
| 29    | bwd_switches_merge      | float | Anzahl Weichen rückwärts (Merge/Crossing) |
| 30    | bwd_target_found_merge  | float | 1, wenn Ziel rückwärts erreicht (Merge/Crossing) |
| 31    | bwd_abort_merge         | float | 1, wenn max_steps rückwärts überschritten (Merge/Crossing) |

#### Start-Block (decision_type == 1)
| Index | Name | Typ | Beschreibung |
|-------|------|-----|--------------|
| 28    | delta_dist_fwd | float | Distanzdifferenz zum Ziel nach dem ersten Schritt (nur Start) |

#### Allgemeine Zusatzfeatures
| Index | Name | Typ | Beschreibung |
|-------|------|-----|--------------|
| 32    | agent_state | float | Aktueller Agentenstatus (enum value) |
| 33    | saved_action | float | Letzte gespeicherte Aktion (falls vorhanden, sonst -1) |

---

**Feature-Befüllung und Speziallogik**

- **decision_type** wird immer gesetzt.
- Die Blöcke für Switch und Merge/Crossing werden nur bei passender Situation befüllt, alle anderen Felder bleiben 0.
- Die Features für jede Richtung werden per rekursiver DFS mit Backtracking berechnet (`_navigate_direction`), inkl. Deadlock- und Cycle-Erkennung.
- Unerreichbare Zellen (`np.inf`) werden durch $2 \times$ aktuelle Distanz ersetzt, NaN/Inf werden durch -1 ersetzt.
- Deadlocks werden erkannt, wenn ein entgegenkommender Agent auf dem Pfad ist.
- **abort**-Flags werden gesetzt, wenn die maximale Schrittzahl (max_steps) überschritten wird.
- **target_found** wird gesetzt, wenn das Ziel auf dem Pfad erreicht wird.

---

**Quelle:**
Alle Feature-Indizes, Bedeutungen und Setzlogik sind exakt aus dem aktuellen Code in [decision_point_observation.py](flatland_solver_policy/example/flatland_rail_env/marl_attention_temporal_observation/decision_point_observation.py) extrahiert. Bei Änderungen im Code ist diese Tabelle zu aktualisieren.

---

## 2.1 Ausführliche Beschreibung: DecisionPointObservation

Die `DecisionPointObservation` ist ein spezialisierter Beobachtungs-Builder für Multi-Agenten-Umgebungen im Flatland-Railway-Setting. Ihr Ziel ist es, für jeden Agenten einen Feature-Vektor zu erzeugen, der die wichtigsten Entscheidungssituationen im Bahnnetz abbildet. Sie ist darauf ausgelegt, die Entscheidungslogik an den drei zentralen Punkten im Flatland-Setting zu unterstützen:

1. **Startpunkt (READY_TO_DEPART):** Soll der Agent das Spielfeld betreten?
2. **Weiche (Switch):** Der Agent steht auf einer Weiche und kann abzweigen – hier ist eine Richtungsentscheidung nötig.
3. **Merge/Crossing (vor einer Weiche):** Der Agent steht eine Zelle vor einer Weiche, kann nicht abzweigen, aber es besteht Konfliktpotenzial (z.B. Überholen, Vorrang).

### Feature-Vektor und Entscheidungslogik

Die Beobachtung besteht aus einem 42-dimensionalen Feature-Vektor (im Code: `self.feature_len`). Die ersten Features kodieren den Entscheidungstyp und geben einen Hinweis auf die beste Richtung (one-hot für links, geradeaus, rechts) entlang des kürzesten Pfads zum Ziel. Die weiteren Features sind in Blöcke für die verschiedenen Entscheidungssituationen unterteilt und werden nur befüllt, wenn die jeweilige Situation vorliegt.

- **decision_type (Feature 0):** Gibt an, in welcher Entscheidungssituation sich der Agent befindet (0 = normal, 1 = Start, 2 = Switch, 4 = Merge/Crossing, 8 = DONE).
- **Richtungshinweis (Features 1-3):** One-hot-Vektor, der die beste Richtung (links, geradeaus, rechts) zum Ziel markiert.
- **Switch-Block (Features 4-21):** Für jede mögliche Richtung (links, geradeaus, rechts, rückwärts) werden Distanz zum Ziel, Deadlock-Flag, Anzahl durchlaufener Weichen, Zielerreichung und Abbruch-Flag berechnet.
- **Merge/Crossing-Block (Features 22-29):** Betrachtet speziell die Situation vor einer Weiche, sowohl vorwärts als auch rückwärts.
- **Agentenstatus (Features 30-36):** One-hot-Kodierung des aktuellen Agentenstatus (READY_TO_DEPART, MALFUNCTION_OFF_MAP, MOVING, STOPPED, MALFUNCTION, DONE).
- **Letzte Aktion (Features 37-41):** One-hot-Kodierung der zuletzt gespeicherten Aktion des Agenten.

### Entscheidungsfindung im Detail

- **Entscheidungstyp-Bestimmung:** Der Code prüft, ob der Agent am Start steht, auf einer Weiche ist oder sich vor einer Weiche befindet. Je nach Situation werden die entsprechenden Feature-Blöcke befüllt.
- **Pfadbewertung:** Für jede relevante Richtung wird per rekursiver Tiefensuche (DFS) der Pfad zum Ziel analysiert. Dabei werden Deadlocks, Zyklen und andere Agenten erkannt. Die DFS ist so gestaltet, dass sie an jedem Switch alle Alternativen ausprobiert, um Deadlocks zu vermeiden.
- **Konflikt- und Deadlockerkennung:** Trifft der Agent auf einen entgegenkommenden Agenten, wird dies als potenzieller Deadlock erkannt und im Feature-Vektor kodiert.
- **Abbruch-Flag:** Wird die maximale Schritttiefe der Suche überschritten, wird ein Abbruch-Flag gesetzt.
- **Zielerreichung:** Wird das Ziel auf dem Pfad gefunden, wird dies ebenfalls im Feature-Vektor markiert.

### Multi-Agenten-Logik

Die Klasse speichert für jeden Agenten, welche gegnerischen Agenten auf dem Pfad begegnet wurden. Diese Information kann für Konfliktlösung und Prioritätsentscheidungen genutzt werden.

### Methodenüberblick

- **`get(handle)`**: Erzeugt den Feature-Vektor für einen einzelnen Agenten.
- **`get_many(handles)`**: Erzeugt die Beobachtungen für mehrere Agenten gleichzeitig.
- **`_shortest_path_action_hint(...)`**: Berechnet, welche Richtung (links, geradeaus, rechts) entlang des kürzesten Pfads zum Ziel optimal ist.
- **`_navigate_direction(...)`**: Führt die rekursive Tiefensuche durch, um Pfadmetriken, Deadlocks und Zielerreichung zu bestimmen.

### Besonderheiten

- Die Features sind so angeordnet, dass sie für RL-Algorithmen direkt nutzbar sind.
- Die Beobachtung ist disjunkt: Nur die für die aktuelle Entscheidungssituation relevanten Features werden befüllt, alle anderen bleiben 0.
- Die Klasse ist darauf ausgelegt, sowohl Einzelagenten- als auch Multi-Agenten-Szenarien effizient zu unterstützen.

---

**Fazit:**  
Die `DecisionPointObservation` abstrahiert die komplexen Entscheidungssituationen im Flatland-Railway-Setting in einen strukturierten, RL-tauglichen Feature-Vektor. Sie erkennt und kodiert alle relevanten Entscheidungs- und Konfliktpunkte, sodass Policies gezielt auf diese Situationen reagieren können.

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