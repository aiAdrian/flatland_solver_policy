#### Erweiterung für Merge/Kreuzung (decision_type==3)

Für Situationen, in denen sich ein Agent eine Zelle vor einer Weiche, Einmündung oder Kreuzung befindet (decision_type==3), wurde die Beobachtung um folgende Logik erweitert:


Diese Erweiterung ermöglicht Policies, an Einmündungen und Kreuzungen situationsabhängig zu entscheiden, ob ein Einfädeln, Kreuzen oder Warten sinnvoll ist, und so Deadlocks und Konflikte proaktiv zu vermeiden.
### Erweiterte DecisionPointObservation (2026)

Die DecisionPointObservation wurde um eine fortschrittliche Navigations- und Deadlock-Logik erweitert, die speziell für Agenten an Weichen entwickelt wurde:

	- Die Distanz zum Ziel (bzw. -1, falls kein Weg gefunden wurde)
	- Ein Deadlock-Flag (1, falls Deadlock, sonst 0)
	- Die Anzahl der Pfadwechsel (wie oft beim Backtracking eine neue Richtung gewählt wurde)

Durch diese Erweiterung ist die DecisionPointObservation besonders mächtig für Multi-Agenten-Szenarien mit komplexen Weichen- und Konfliktsituationen.
# Übersicht der vier Observations in `temporal_multi_agent_observation`

In diesem Modul werden vier verschiedene Beobachtungsarten (Observations) für Multi-Agenten-Umgebungen im Flatland-Rail-Setting implementiert. Im Folgenden werden diese vier Observations in Prosa beschrieben:

## Basisklassen und Alternativen

Neben den temporalen Observations gibt es drei wichtige Basisklassen, die als Grundlage oder Alternative für die zeitlichen Varianten dienen:


### 1. `ExperimentalObservation`
Diese Beobachtung liefert eine umfassende, 30-dimensionale Feature-Vector-Repräsentation für jeden Agenten. Sie kombiniert eigene Agenten-Informationen (Position, Richtung, Ziel, Status) mit einer Analyse der Umgebung und anderer Agenten. Die wichtigsten Merkmale:


Diese Observation ist besonders geeignet als Basis für komplexere, temporale oder multi-agentenfähige Beobachtungen.



### 2. `DecisionPointObservation`

Diese Observation ist speziell auf die drei wichtigsten Entscheidungssituationen im Flatland-Setting zugeschnitten:

1. **Start (READY_TO_DEPART):** Der Agent entscheidet, ob er auf das Spielfeld fährt.
2. **An einer Weiche:** Der Agent steht auf einer Weiche und kann abzweigen (klassische Routing-Entscheidung, mehrere Pfade möglich).
3. **Vor einer Weiche (Merge/Crossing):** Der Agent steht ein Feld vor einer Weiche, kann nicht abzweigen, aber es besteht Konfliktpotenzial durch Einfädeln oder Kreuzen.

**Feature-Design und Logik:**

Die Feature-Vektoren sind für alle Entscheidungstypen disjunkt belegt:

| Index    | Bedeutung (decision_type==2, Weiche)         | Bedeutung (decision_type==3, Merge/Kreuzung) |
|----------|----------------------------------------------|----------------------------------------------|
| 0        | decision_type                                | decision_type                                |
| 1–3      | one-hot shortest path hint [l, f, r]         | one-hot shortest path hint [l, f, r]         |
| 4–7      | left: dist, deadlock, switches, delta_dist   | –                                            |
| 8–11     | forward: dist, deadlock, switches, delta_dist| –                                            |
| 12–15    | right: dist, deadlock, switches, delta_dist  | –                                            |
| 16–19    | reverse: dist, deadlock, switches, delta_dist| –                                            |
| 20–23    | –                                           | forward: dist, deadlock, switches, delta_dist |
| 24–27    | –                                           | backward: dist, deadlock, switches, delta_dist|
| 28       | delta_dist_fwd (decision_type==1)            | delta_dist_fwd (decision_type==1)            |

**Speziallogik für Weichen (decision_type==2):**
- Für jede Richtung (links, geradeaus, rechts, rückwärts) werden vier Werte gespeichert: dist (Distanz zum Ziel), deadlock (1/0), switches (Pfadwechsel), delta_dist (Distanzdifferenz zum Ziel).
- Die Blöcke sind im Vektor disjunkt: left (4–7), forward (8–11), right (12–15), reverse (16–19).

**Speziallogik für Merge/Kreuzung (decision_type==3):**
- Für forward (20–23) und backward (24–27) werden jeweils dist, deadlock, switches, delta_dist gespeichert.
- Die Werte sind analog zu type==2.

**Zusammenfassung:**


### 3. `SimplifiedPathThreeTierObservation`
Diese Beobachtung teilt die Umgebung eines Agenten in drei Pfadsegmente ("Tiers") auf:


**Feature-Design:**

**Besonderheiten:**

## 1. `TemporalMultiAgentObservation`
Diese Observation kombiniert die Beobachtungen mehrerer Agenten über verschiedene Zeitschritte hinweg. Jeder Agent erhält nicht nur die aktuelle Umgebung, sondern auch eine Historie seiner eigenen Beobachtungen. Dadurch kann der Agent zeitliche Zusammenhänge und Veränderungen in der Umgebung besser erfassen und für seine Entscheidungsfindung nutzen. Die Beobachtung ist besonders nützlich für Algorithmen, die auf zeitlichen Abhängigkeiten basieren, wie z.B. rekurrente neuronale Netze.

## 2. `TemporalMultiAgentSwitchObservation`
Hierbei handelt es sich um eine spezialisierte Variante der temporalen Multi-Agenten-Observation, die den Fokus auf Weichen (Switches) im Schienennetz legt. Die Beobachtung enthält explizite Informationen über die Positionen und Zustände von Weichen, sowohl für den aktuellen als auch für vergangene Zeitschritte. Dies ermöglicht es den Agenten, strategische Entscheidungen an kritischen Knotenpunkten im Schienennetz zu treffen, insbesondere im Hinblick auf Konfliktvermeidung und effiziente Routenwahl.

## 3. `TemporalMultiAgentSwitchCellObservation`
Diese Observation erweitert die vorherige, indem sie nicht nur die Weichen selbst, sondern auch die Zellen, die zu Weichen führen oder in deren Nähe liegen, explizit berücksichtigt. Die Agenten erhalten Informationen darüber, ob sie sich auf einer Weichenzelle befinden oder sich einer solchen nähern. Dies ist besonders relevant für das Warten vor Weichen, das Überholen und die Lösung von Konflikten zwischen Agenten.

## 4. `TemporalMultiAgentSwitchCellWithDirectionObservation`
Diese Beobachtung ergänzt die `SwitchCellObservation` um Richtungsinformationen. Die Agenten erhalten nicht nur die Information, ob sie sich auf einer Weichenzelle oder in deren Nähe befinden, sondern auch, in welche Richtung sie sich bewegen und welche Abzweigungen in ihrer aktuellen Richtung möglich sind. Dadurch können Agenten noch gezielter planen, wann und wie sie an Weichen abbiegen oder warten sollten, um Kollisionen und Staus zu vermeiden.


Jede dieser Observations ist darauf ausgelegt, die Entscheidungsfindung der Agenten in komplexen, dynamischen Schienennetzen zu verbessern, indem sie relevante Umgebungsinformationen über die Zeit und im Kontext von Weichen bereitstellt.